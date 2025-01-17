import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import json
import os
from PIL import Image
from torchvision.transforms import ToTensor
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import *
img_to_tensor = ToTensor()

# fixing HTTPS issue on Colab
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


class MiniImageNetDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        tags_names = pd.read_csv('/content/gdrive/MyDrive/TenGrad/mini_imagenet_class_label_dict3.txt', sep=" ", header=None)
        tags_names.columns = ["label", "number", "name"]
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.tags_names = tags_names

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0][0:9],
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        image = Image.fromarray(image)
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        target = self.landmarks_frame.iloc[idx, 0][0:9]
        sample = image, self.landmarks_frame.iloc[idx, 0][0:9]
        target = int(self.tags_names[self.tags_names['label'] == target]["number"]) - 1
        if self.transform is not None:
            image = self.transform(image)

        return image, target


class TestingDataset(Dataset):
  def __init__(self, list='/content/IMagenet/testing_data_list.json'):
    with open(list, 'r') as f:
      self.list = json.load(f)

  def __getitem__(self, index):
    img_dir, label = self.list[index]
    img = Image.open(img_dir)
    tensor = img_to_tensor(img)
    return tensor, label

  def __len__(self):
    return len(self.list)



class TrainingDataset(Dataset):
  def __init__(self, list='/content/IMagenet/training_data_list.json'):
    with open(list, 'r') as f:
      self.list = json.load(f)

  def __getitem__(self, index):
    img_dir, label = self.list[index]
    img = Image.open(img_dir)
    tensor = img_to_tensor(img)
    return tensor, label

  def __len__(self):
    return len(self.list)

def get_transforms(dataset):
    transform_train = None
    transform_test = None

    if dataset == 'imagenet':
        transform_train = transforms.Compose([
            #transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])        
        transform_test = transforms.Compose([
            #transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])    

    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    if dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    if dataset == 'fashion-mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2862,), (0.3529,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2862,), (0.3529,))
        ])

    assert transform_test is not None and transform_train is not None, 'Error, no dataset %s' % dataset
    return transform_train, transform_test


def get_dataloader(dataset, train_batch_size, test_batch_size, num_workers=2, root='../data', csv_dir=None):
    transform_train, transform_test = get_transforms(dataset)
    trainset, testset = None, None
    if dataset == 'imagenet':
        '''
        train_mini_imagnet = pd.read_csv(csv_dir+'/train.csv')
        test_mini_imagnet = pd.read_csv(csv_dir+'/test.csv')
        trainset = MiniImageNetDataset(csv_file=csv_dir+'/train.csv',
                                    root_dir='/content/gdrive/MyDrive/mini-imagenet/train', transform=transform_train)
        testset = MiniImageNetDataset(csv_file=csv_dir+'/test.csv',
                                    root_dir='/content/gdrive/MyDrive/mini-imagenet/test', transform=transform_test)
        '''
        train_dir = csv_dir + '/train'
        trainset = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
        test_dir = csv_dir + '/val/validation-folders'
        testset = datasets.ImageFolder(test_dir, transform=transforms.ToTensor())
        '''
        training_data_list = []
        class_name = 0
        targets = {}
        for folder in os.listdir('/content/IMagenet/tiny-imagenet-200/train/'):
            label = class_name  # The name of the folder is the label of the images it contains
            for file in os.listdir('/content/IMagenet/tiny-imagenet-200/train/' + folder + '/images/'):
                file_dir = '/content/IMagenet/tiny-imagenet-200/train/' + folder + '/images/' + file
                training_data_list.append((file_dir, label))
            class_name = class_name + 1
            targets.update({folder: class_name})
        with open('./training_data_list.json', 'w') as f:
            json.dump(training_data_list, f)
            
        testing_data_list = []
        with open('/content/IMagenet/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
            for line in f.readlines():
                file, label = line.split()[0:2]
                label = targets[label]
                file_dir = '/content/IMagenet/tiny-imagenet-200/val/images/' + file
                testing_data_list.append((file_dir, label))

        with open('/content/IMagenet/testing_data_list.json', 'w') as f:
            json.dump(testing_data_list, f)
        '''
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform_test)

    assert trainset is not None and testset is not None, 'Error, no dataset %s' % dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers, drop_last=True)

    return trainloader, testloader