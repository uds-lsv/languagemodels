import torch


def repackage_hidden(hidden_state):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach()
    else:
        return tuple(repackage_hidden(t) for t in hidden_state)