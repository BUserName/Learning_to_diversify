from models import caffenet
# from models import mnist
# from models import patch_based
# from models import alexnet

nets_map = {
    'caffenet': caffenet.caffenet,
}

def get_network(name):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_network_fn
