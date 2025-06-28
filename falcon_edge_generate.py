import os
import numpy as np
import coremltools as ct
import time
from transformers import AutoTokenizer
import shutil
from argparse import ArgumentParser


def copy_compiled_model(mlmodel: ct.models.MLModel, dest: str):
    compiled_model_path = mlmodel.get_compiled_model_path()
    shutil.copytree(compiled_model_path, dest, dirs_exist_ok=True)


def load_mlmodel(path, function_name, copy_compiled):
    extension = os.path.splitext(path)[1]
    if extension == ".mlmodelc":
        return ct.models.CompiledMLModel(
            path,
            function_name=function_name,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
    else:
        mlmodel = ct.models.MLModel(
            path,
            function_name=function_name,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        if copy_compiled:
            copy_compiled_model(mlmodel, path.replace(".mlpackage", ".mlmodelc"))
        return mlmodel


def load_embeddings(path):
    return np.load(path)


class ModelContainer:
    def __init__(
        self,
        embeddings_path,
        mlmodel_path,
        lm_head_path,
        cache_length,
        hf_model,
        temp=0.7,
        min_p=0.1,
    ):
        self.mlmodel_path = mlmodel_path
        self.embeddings_path = embeddings_path
        self.lm_head_path = lm_head_path
        self.cache_length = cache_length
        self.temp = temp
        self.min_p = min_p
        print("Loading embeddings...")
        self.embeddings = load_embeddings(embeddings_path)
        print("Loading generation model...")
        self.generation_model = load_mlmodel(
            mlmodel_path, f"model_input_1_cache_{cache_length}", copy_compiled=True
        )
        # self.prompt_model = None
        print("Loading prompt model...")
        self.prompt_model = load_mlmodel(
            mlmodel_path.replace(".mlpackage", ".mlmodelc"),
            f"model_input_64_cache_{cache_length}",
            copy_compiled=False,
        )
        print("Loading lm head model...")
        self.lm_head_model = load_mlmodel(
            lm_head_path,
            "min_p_length_1" if temp > 0 else "lm_head_length_1",
            copy_compiled=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.end_of_response_token_id = self.tokenizer("<|im_end|>").input_ids[0]

        self.state = None
        self.position = None
        self.attention_mask = None

    def initialize_generation(self):
        self.state = self.generation_model.make_state()
        attention_mask = np.arange(self.cache_length, dtype=np.int32)
        attention_mask = attention_mask[:, None] >= attention_mask[None, :]
        attention_mask = attention_mask[None, None, :, :]
        self.attention_mask = np.where(
            attention_mask,
            np.array(0.0, dtype=np.float16),
            np.array(-np.inf, dtype=np.float16),
        )
        self.position = 0

    def load_prompt_model(self):
        if self.prompt_model is None:
            self.prompt_model = load_mlmodel(
                self.mlmodel_path,
                f"model_input_64_cache_{self.cache_length}",
                copy_compiled=False,
            )

    def unload_prompt_model(self):
        del self.prompt_model
        self.prompt_model = None

    def embed(self, ids):
        return self.embeddings[ids]  # .transpose(0, 2, 1)  # [..., None, :]

    def process_prompt(self, prompt):
        if self.prompt_model is None:
            self.load_prompt_model()
        messages = [{"role": "user", "content": prompt}]
        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        stop_processing = False
        start_time = time.perf_counter()
        processed_chunks = 0
        for i in range(0, len(tokens), 64):
            chunk = tokens[i : min(i + 64, len(tokens))]
            if self.position + len(chunk) > self.cache_length:
                stop_processing = True
                break
            processed_chunks += 1
            embds = self.embed([chunk]).transpose(0, 2, 1)[
                ..., None, :
            ]  # [..., None, :]
            if len(chunk) < 64:
                embds = np.concat(
                    (
                        embds,
                        np.zeros(
                            (1, embds.shape[1], 1, 64 - len(chunk)), dtype=np.float16
                        ),
                    ),
                    axis=-1,
                )
            kv_write_idx = np.array([self.position], dtype=np.int32)
            positions = np.arange(self.position, self.position + 64, dtype=np.int32)[
                None, :
            ]
            attention_mask = self.attention_mask[
                :, :, self.position : self.position + 64
            ]
            pred = self.prompt_model.predict(
                {
                    "hidden_states": embds,
                    "kv_write_idx": kv_write_idx,
                    "positions": positions,
                    "attention_mask": attention_mask,
                },
                self.state,
            )
            self.position += len(chunk)
        self.unload_prompt_model()
        end_time = time.perf_counter()
        print(
            f"==== Processed {processed_chunks * 64} tokens in {end_time - start_time:.2f} seconds, {processed_chunks * 64 / (end_time - start_time):.2f} tokens per second, current position: {self.position}",
        )
        if stop_processing:
            return np.array([-1], dtype=np.int32)
        output_hidden_states = pred["output_hidden_states"][..., [len(chunk) - 1]]
        return self.lm_head(output_hidden_states)

    def lm_head(self, hidden_states):
        if self.temp > 0:
            input_id = self.lm_head_model.predict(
                {
                    "hidden_states": hidden_states,
                    "temp": np.array([self.temp], dtype=np.float16),
                    "p": np.array([self.min_p], dtype=np.float16),
                    "random_number": np.random.uniform(0.0, 1.0, (1,)),
                }
            )["sampled_index"][:, 0]
        else:
            input_id = self.lm_head_model.predict(
                {
                    "hidden_states": hidden_states,
                }
            )[
                "argmax"
            ][:, 0]
        return input_id

    def generate(self, input_id: np.array):
        stop_generation = False
        # for i in range(max_new_tokens):
        start_time = time.perf_counter()
        generated_tokens = 0
        while self.position < self.cache_length:
            generated_tokens += 1
            embd = self.embed(input_id).transpose(0, 3, 1, 2)
            hidden_states = self.generation_model.predict(
                {
                    "hidden_states": embd,
                    "kv_write_idx": np.array([self.position], dtype=np.int32),
                    "positions": np.array([[self.position]], dtype=np.int32),
                    "attention_mask": self.attention_mask[:, :, [self.position]],
                },
                self.state,
            )["output_hidden_states"]
            if stop_generation:
                print()
                # print("Loading prompt model...")
                self.position += 1
                break

            input_id = self.lm_head(hidden_states)

            input_id_item = input_id.item()
            if input_id_item == self.end_of_response_token_id:
                stop_generation = True
            print(self.tokenizer.decode(input_id_item), end="", flush=True)
            self.position += 1

        end_time = time.perf_counter()
        print(
            f"==== Generated {generated_tokens} tokens in {end_time - start_time:.2f} seconds, {generated_tokens / (end_time - start_time):.2f} tokens per second, current position: {self.position}",
        )
        # if stop_generation:
        #     self.load_prompt_model()

    def loop(self):
        self.initialize_generation()
        print("Begin conversation...")
        while True:
            print(">>> ", end="", flush=True)
            self.load_prompt_model()
            prompt = input()
            prompt_result = self.process_prompt(prompt)
            if prompt_result.item() == -1:
                print("\n--- END OF CONVERSATION: MAX CONTEXT LENGTH REACHED ---\n")
                break
            print(self.tokenizer.decode(prompt_result.item()), end="", flush=True)
            self.generate(prompt_result)
            if self.position >= (self.cache_length):
                print("\n--- END OF CONVERSATION: MAX CONTEXT LENGTH REACHED ---\n")
                break


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lm_head", type=str, required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument(
        "--cache_length",
        type=int,
        choices=[1024, 2048, 2048 + 1024, 4096, 4096 + 2048, 8192],
        default=1024,
    )
    parser.add_argument("--min_p", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.7)
    # parser.add_argument("--hf_model", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()
    ModelContainer(
        args.embeddings,
        args.model,
        args.lm_head,
        args.cache_length,
        "tiiuae/Falcon-E-1B-Instruct",
        args.temp,
        args.min_p,
    ).loop()


if __name__ == "__main__":
    main()
