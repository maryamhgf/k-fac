from models.cifar import (alexnet, densenet, resnet,
                          vgg16_bn, vgg19_bn, vgg16, vgg13,
                          wrn, inception, googlenet, xception, nasnet, mobilenetv2)
from models.mnist import (fc, convnet, bn, toy, autoencoder)
from models.imagenet import (resnext)


def get_network(network, **kwargs):
    networks = {
        'toy': toy,
        'bn': bn,
        'fc': fc,
        'convnet': convnet,
        'alexnet': alexnet,
        'densenet': densenet,
        'resnet': resnet,
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'vgg16': vgg16,
        'vgg13': vgg13,
        'wrn': wrn,
        'inception': inception,
        "googlenet": googlenet,
        "xception": xception,
        "nasnet": nasnet,
        "resnext": resnext,
        "mobilenetv2": mobilenetv2,
        "autoencoder": autoencoder

    }

    return networks[network](**kwargs)

