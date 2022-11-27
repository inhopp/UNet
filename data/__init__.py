from .dataset import Dataset
import torch
import torchvision.transforms as transforms

def generate_loader(phase, opt):
    dataset = Dataset
    img_size = opt.input_size

    if phase == 'train':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    dataset = dataset(opt, phase, transform=transform)

    kwargs = {
        "batch_size": opt.batch_size if phase == 'train' else opt.eval_batch_size,
        "shuffle": phase == 'train',
        "drop_last": phase == 'train',
    }

    return torch.utils.data.DataLoader(dataset, **kwargs)