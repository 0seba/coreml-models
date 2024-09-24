import numpy as np
import coremltools.converters.mil as mil
from coremltools.converters.mil import Builder as mb, Var

from .utils_c import *
from .choices import *

def topK_genrate_mil(hidden_states, input_ids, head, logits_processor, total_tokens, depth, top_k, threshold):
    """
    CoreML MIL implementation of the topK_genrate method.

    Args:
        hidden_states: The hidden states tensor.
        input_ids: The input ids tensor.
        head: The head layer.
        logits_processor: The logits processor.
        total_tokens: The total number of tokens to generate.
        depth: The depth of the tree.
        top_k: The number of top-k candidates to consider.
        threshold: The threshold for pruning the tree.

    Returns:
        A tuple containing the draft tokens, retrieve indices, tree mask, and tree position ids.
    """

    # Convert input tensors to MIL variables
    hidden_states = mb.const(val=hidden_states, name="hidden_states")
    input_ids = mb.const(val=input_ids, name="input_ids")

    # Extract relevant parameters
    device = hidden_states.dtype.device
    len_posi = input_ids.shape[1]
    tree_mask_init = mb.const(val=np.eye(top_k, dtype=np.bool), name="tree_mask_init")
    position_ids = mb.const(val=np.zeros(top_k, dtype=np.int32), name="position_ids")
    topk_cs_index = mb.const(val=np.arange(top_k, dtype=np.int32), name="topk_cs_index")

    # Initialize variables
    scores_list = []
    parents_list = []
    ss_token = []
    tree_mask = tree_mask_init

    # Process the initial input
    input_ids = input_ids[:, 1:]
    out_hidden, past_key_values = model_forward_mil(hidden_states, input_ids, past_key_values=None, use_cache=True)
    last_hidden = mb.gather(x=out_hidden, indices=mb.const(val=np.array([-1]), name="last_hidden_index"), axis=1)
    last_headout = head(last_hidden)
    last_p = mb.softmax(x=last_headout, axis=-1)
    topk_index, topk_p = mb.topk(x=last_p, k=top_k, axis=-1)
    scores = mb.gather(x=topk_p, indices=mb.const(val=np.array([0]), name="scores_index"), axis=0)
    scores_list.append(scores[None])
    parents_list.append(mb.const(val=np.zeros(1, dtype=np.int32), name="parents_init"))
    ss_token.append(topk_index)
    input_ids = topk_index
    input_hidden = mb.expand_dims(x=last_hidden, axes=[0], name="expand_last_hidden")
    input_hidden = mb.repeat(x=input_hidden, repeats=top_k, axis=1, name="repeat_last_hidden")
    tree_mask = mb.expand_dims(x=tree_mask, axes=[0, 0], name="expand_tree_mask")

    # Iterate through the tree depth
    for i in range(depth):
        # Update position ids
        position_ids = mb.add(x=position_ids, y=mb.const(val=len_posi + i, name="position_ids_update"), name="position_ids_add")

        # Forward pass through the model
        out_hidden, past_key_values = model_forward_mil(input_hidden, input_ids, past_key_values=past_key_values, position_ids=position_ids, use_cache=True)

        # Calculate parents
        bias1 = mb.const(val=top_k if i > 0 else 0, name="bias1")
        bias2 = mb.const(val=max(0, i - 1), name="bias2")
        bias = mb.const(val=1, name="bias_const")
        bias = mb.add(x=bias, y=mb.mul(x=mb.const(val=top_k ** 2, name="topk_squared"), y=bias2, name="mul_bias2"), name="add_bias2")
        bias = mb.add(x=bias, y=bias1, name="add_bias1")
        parents = mb.add(x=topk_cs_index, y=bias, name="parents_add")
        parents_list.append(parents)

        # Calculate scores
        last_headout = head(out_hidden[0])
        last_p = mb.softmax(x=last_headout, axis=-1)
        topk_index, topk_p = mb.topk(x=last_p, k=top_k, axis=-1)
        cu_scores = mb.add(x=topk_p, y=mb.expand_dims(x=scores, axes=[1], name="expand_scores"), name="cu_scores_add")

        # Select top-k candidates
        topk_cs_index, topk_cs_p = mb.topk(x=cu_scores.reshape(-1), k=top_k, axis=-1)
        scores = topk_cs_p
        out_ids = mb.floor_div(x=topk_cs_index, y=mb.const(val=top_k, name="topk_const"), name="out_ids_div")
        input_hidden = mb.gather(x=out_hidden, indices=out_ids, axis=1, name="gather_input_hidden")
        input_ids = mb.gather(x=topk_index.reshape(-1), indices=topk_cs_index, axis=0, name="gather_input_ids")
        input_ids = mb.expand_dims(x=input_ids, axes=[0], name="expand_input_ids")

        # Update lists
        ss_token.append(topk_index)
        scores_list.append(cu_scores)
        tree_mask = mb.concat(values=(tree_mask[:, :, out_ids], tree_mask_init), axis=3, name="concat_tree_mask")

        # Check for threshold condition
        # if threshold < 0 and cu_scores.max() < threshold:
        #     break

    # Concatenate scores and tokens
    scores_list = mb.concat(values=scores_list, axis=0, name="concat_scores_list")
    scores_list = mb.reshape(x=scores_list, shape=(-1,), name="reshape_scores_list")
    ss_token_list = mb.concat(values=ss_token, axis=0, name="concat_ss_token_list")
    ss_token_list = mb.reshape(x=ss_token_list, shape=(-1,), name="reshape_ss_token_list")

    # Select top tokens
    top_scores_index = mb.topk(x=scores_list, k=total_tokens, axis=-1).indices
    top_scores_index = mb.sort(x=top_scores_index).values

    # Gather draft tokens
    draft_tokens = mb.gather(x=ss_token_list, indices=top_scores_index, axis=0, name="gather_draft_tokens")
    draft_tokens = mb.concat(values=(mb.const(val=np.array([input_ids[0, 0]]), name="sample_token"), draft_tokens), axis=0, name="concat_draft_tokens")

    # Gather parents
    draft_parents = mb.gather(x=mb.concat(values=parents_list, axis=0, name="concat_parents_list"), indices=mb.floor_div(x=top_scores_index, y=mb.const(val=top_k, name="topk_const"), name="parents_div"), axis=0, name="gather_draft_parents")

    # Calculate mask index
    mask_index = mb.searchsorted(x=top_scores_index, y=mb.sub(x=draft_parents, y=mb.const(val=1, name="one_const"), name="parents_sub"), side="left", name="searchsorted_mask_index")
    mask_index = mb.add(x=mask_index, y=mb.const(val=1, name="one_const"), name="mask_index_add")
    mask_index = mb.where(condition=mb.equal(x=draft_parents, y=mb.const(val=0, name="zero_const"), name="parents_eq"), x=mb.const(val=-1, name="neg_one_const"), y=mask_index, name="where_mask_index")

    # Create tree mask
    tree_mask = mb.const(val=np.eye(total_tokens + 1, dtype=np.bool), name="tree_mask_const")
    tree_mask = mb.set_item(x=tree_mask, indices=mb.const(val=np.array([0]), name="zero_index"), value=mb.const(val=True, name="true_const"), name="set_item_tree_mask")
    mask_index_list = mb.to_list(x=mask_index, name="to_list_mask_index")
    for i in range(total_tokens):
        tree_mask = mb.set_item(x=tree_mask, indices=mb.const(val=np.array([i + 1]), name="index_add"), value=mb.add(x=tree_mask[mask_index_list[i]], y=tree_mask[i + 1], name="add_tree_mask"), name="set_item_tree_mask")

    # Calculate tree position ids
    tree_position_ids = mb.reduce_sum(x=tree_mask, axes=[1], keep_dims=False, name="reduce_sum_tree_position_ids")
    tree_position_ids = mb.sub(x=tree_position_ids, y=mb.const(val=1, name="one_const"), name="tree_position_ids_sub")

    # Convert tree mask to float
    tree_mask = mb.cast(x=tree_mask, dtype="fp32", name="cast_tree_mask")
    tree_mask = mb.expand_dims(x=tree_mask, axes=[0, 0], name="expand_tree_mask")

    # Convert draft tokens to tensor
    draft_tokens = mb.expand_dims(x=draft_tokens, axes=[0], name="expand_draft_tokens")

    # Calculate retrieve indices
    max_depth = mb.reduce_max(x=tree_position_ids, axes=[0], keep_dims=False, name="reduce_max_max_depth")
    max_depth = mb.add(x=max_depth, y=mb.const(val=1, name="one_const"), name="max_depth_add")
    noleaf_index = mb.unique(x=mask_index, name="unique_noleaf_index")
    noleaf_num = mb.sub(x=mb.shape(x=noleaf_index, name="shape_noleaf_index")[0], y=mb.const(val=1, name="one_const"), name="noleaf_num_sub")
    leaf_num = mb.sub(x=total_tokens, y=noleaf_num, name="leaf_num_sub")
    retrieve_indices = mb.const(val=np.zeros((leaf_num.val, max_depth.val), dtype=np.int32) - 1, name="retrieve_indices_init")
    retrieve_indices = mb.to_list(x=retrieve_indices, name="to_list_retrieve_indices")
    position_ids_list = mb.to_list(x=tree_position_ids, name="to_list_position_ids")
    rid = mb.const(val=0, name="rid_init")
    for i in range(total_tokens + 1):
        if i not in noleaf_index.val:
            cid = mb.const(val=i, name=f"cid_{i}")
            depth = mb.const(val=position_ids_list[i], name=f"depth_{i}")
            for j in reversed(range(depth.val + 1)):
                retrieve_indices[rid.val][j] = cid.val
                cid = mb.const(val=mask_index_list[cid.val - 1], name=f"cid_update_{i}_{j}")
            rid = mb.add(x=rid, y=mb.const(val=1, name="one_const"), name="rid_update")

    # Sort retrieve indices
    retrieve_indices = mb.sort(x=retrieve_indices, axis=0, name="sort_retrieve_indices").values

    # Convert retrieve indices to tensor
    retrieve_indices = mb.const(val=retrieve_indices, name="retrieve_indices_const")

    # Return results
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

def model_forward_mil(hidden_states, input_ids, past_key_values=None, position_ids=None, use_cache=True):
    """
    CoreML MIL implementation of the model forward pass.

    Args:
        hidden_states: The hidden states tensor.
        input_ids: The input ids tensor.
        past_key_values: The past key values.
        position_ids: The position ids.
        use_cache: Whether to use cache.

    Returns:
        A tuple containing the output hidden states and past key values.
    """

    # Convert input tensors to MIL variables
    hidden_states = mb.const(val=hidden_states, name="hidden_states")
    input_ids = mb.const(val=input_ids, name="input_ids")
    if position_ids is not None:
        position_ids = mb.const(val=position_ids, name="position_ids")

    # Forward pass through the model
    out_hidden = model(hidden_states=hidden_states, input_ids=input_ids, past_key_values=past_key_values, position_ids=position_ids, use_cache=use_cache)

    # Return results
    return out_hidden, past_key_values

