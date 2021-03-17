from models.cifar import (alexnet, densenet, resnet,
                          vgg16_bn, vgg19_bn,
                          wrn)
from models.mnist import (fc, convnet, bn)


def get_network(network, **kwargs):
    networks = {
        'bn': bn,
        'fc': fc,
        'convnet': convnet,
        'alexnet': alexnet,
        'densenet': densenet,
        'resnet': resnet,
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'wrn': wrn

    }

    return networks[network](**kwargs)

