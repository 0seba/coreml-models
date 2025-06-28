You can convert the Falcon Edge models with the following command

`python -m src.models.falcon_edge_bitnet.convert --model_path ~/.cache/huggingface/hub/models--tiiuae--Falcon-E-3B-Instruct/snapshots/958cabcf7c8e5abbcf94cb0d7cd19be6a5e6f8ee/ --convert_model --output_name falcon_edge_3b --convert_lm_head --export_embeddings`