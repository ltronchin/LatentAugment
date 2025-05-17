import pickle
import torch
import pickle

def load_network_pkl(f):
    data = _LegacyUnpickler(f).load()

    # Add missing fields.
    if 'training_set_kwargs' not in data:
        data['training_set_kwargs'] = None
    if 'augment_pipe' not in data:
        data['augment_pipe'] = None

    # Validate contents.
    assert isinstance(data['G'], torch.nn.Module)
    assert isinstance(data['D'], torch.nn.Module)
    assert isinstance(data['G_ema'], torch.nn.Module)
    assert isinstance(data['training_set_kwargs'], (dict, type(None)))
    assert isinstance(data['augment_pipe'], (torch.nn.Module, type(None)))

    return data

class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module, name)

