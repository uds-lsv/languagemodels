import torch


def repackage_hidden(hidden_state):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach()
    else:
        return tuple(repackage_hidden(t) for t in hidden_state)


def get_num_occurrences_in_tensor(value: int, t: torch.Tensor) -> int:
    word_ids, word_counts = torch.unique(t, return_counts=True)
    value_count = (word_counts == value).nonzero(as_tuple=True)
    if value_count.numel() == 0:
        return 0
    return value_count.item()