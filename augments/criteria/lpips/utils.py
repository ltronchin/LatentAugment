from collections import OrderedDict

import torch


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # build url
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
        + f'master/lpips/weights/v{version}/{net_type}.pth'

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True,
        map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict

def get_state_dict_custom(net_type: str = 'alex', version: str = '0.1', nlin: int = 3):

    # Load the default state.
    state_dict = get_state_dict(net_type, version)
    if nlin == len(state_dict):
        return state_dict

    skip_weights = []
    for i in range(nlin-1):
        skip_weights.append(f'{i}.1.weight')

    new_state_dict = OrderedDict()
    for i, (key, val) in enumerate(state_dict.items()):
        if key in skip_weights:
            continue
        else:
            new_key = key
            new_key = new_key.replace(f'{i}', f'{i - len(skip_weights)}')
            new_state_dict[new_key] = val

    return new_state_dict
