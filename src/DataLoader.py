from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def dataLoader(data='MNIST', batch_size=32):
    #MNIST
    train_dataset = datasets.MNIST('./MNIST/' if data == 'cifar10' else './cifar10/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST('./MNIST/' if data == 'cifar10' else './cifar10/', train=False, transform=transforms.ToTensor(), download=True)

    #DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    return train_loader, test_loader