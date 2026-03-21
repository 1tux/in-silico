from __future__ import annotations


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "encoder") and hasattr(model.transformer.encoder, "layers"):
        return model.transformer.encoder.layers
    raise AttributeError("Unsupported model architecture: could not find decoder layers")


def get_layout(model) -> str:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "batch_first"
    if hasattr(model, "transformer") and hasattr(model.transformer, "encoder") and hasattr(model.transformer.encoder, "layers"):
        return "seq_first"
    raise AttributeError("Unsupported model architecture: could not infer hidden-state layout")


def get_mlp_output_module(layer):
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise AttributeError("Unsupported layer layout: missing mlp module")
    if hasattr(mlp, "down_proj"):
        return mlp.down_proj
    if hasattr(mlp, "dense_4h_to_h"):
        return mlp.dense_4h_to_h
    raise AttributeError("Unsupported MLP layout: missing output projection")


def get_attn_output_module(layer):
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
        return layer.self_attn.o_proj
    if hasattr(layer, "self_attention") and hasattr(layer.self_attention, "dense"):
        return layer.self_attention.dense
    raise AttributeError("Unsupported attention layout: missing output projection")


def get_mlp_input_proxy(model, layer_idx: int):
    layers = get_layers(model)
    return get_mlp_output_module(layers[layer_idx]).input


def get_token_vector_proxy(model, layer_idx: int, token_pos: int):
    proxy = get_mlp_input_proxy(model, layer_idx)
    layout = get_layout(model)
    if layout == "batch_first":
        return proxy[0, token_pos, :]
    return proxy[token_pos, 0, :]


def get_token_neuron_proxy(model, layer_idx: int, token_pos: int, neuron_idx: int):
    proxy = get_mlp_input_proxy(model, layer_idx)
    layout = get_layout(model)
    if layout == "batch_first":
        return proxy[0, token_pos, neuron_idx]
    return proxy[token_pos, 0, neuron_idx]


def get_sequence_neuron_proxy(model, layer_idx: int, neuron_idx: int):
    proxy = get_mlp_input_proxy(model, layer_idx)
    layout = get_layout(model)
    if layout == "batch_first":
        return proxy[0, :, neuron_idx]
    return proxy[:, 0, neuron_idx]


def assign_sequence_neuron(model, layer_idx: int, neuron_idx: int, value):
    proxy = get_mlp_input_proxy(model, layer_idx)
    layout = get_layout(model)
    if layout == "batch_first":
        proxy[0, :, neuron_idx] = value
    else:
        proxy[:, 0, neuron_idx] = value


def assign_token_vector(model, layer_idx: int, token_pos: int, value):
    proxy = get_mlp_input_proxy(model, layer_idx)
    layout = get_layout(model)
    if layout == "batch_first":
        proxy[0, token_pos, :] = value
    else:
        proxy[token_pos, 0, :] = value


def assign_token_neuron(model, layer_idx: int, token_pos: int, neuron_idx: int, value):
    proxy = get_mlp_input_proxy(model, layer_idx)
    layout = get_layout(model)
    if layout == "batch_first":
        proxy[0, token_pos, neuron_idx] = value
    else:
        proxy[token_pos, 0, neuron_idx] = value
