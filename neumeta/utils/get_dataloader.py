from torchvision import transforms, datasets
from torch.utils.data import DataLoader, DistributedSampler


def get_imagenet(batch_size, strong_transform=False, data_dir='/path/to/imagenet', ddp=False):
    
    """
    Returns train and validation data loaders for the ImageNet dataset.

    Args:
        batch_size (int): Number of samples per batch to load.
        strong_transform (bool/str): Indicates whether to apply stronger augmentations.

    Returns:
        tuple: A tuple of train and validation data loaders.
    """
    # Note: ImageNet dataset has higher resolution, i.e., 224 x 224 for typical models like ResNet.
    # You'll also need to provide the path to your ImageNet data by replacing '/path/to/imagenet'.

    # Transforms for the ImageNet dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    # Define stronger augmentation strategies for training set
    if strong_transform:
        if strong_transform == 'v1':
            print("Using strong transform v1")
            # Adjust transformations for higher resolution
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(0, shear=15, scale=(0.8,1.2)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                normalize,
            ])
        # Add other strong transform cases here as per your CIFAR-10 example
        # ...
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    # For ImageNet, generally validation set just uses resizing and normalization
    # Validation transformations
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Load ImageNet training and validation sets
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=val_transforms)

    # DDP: Use DistributedSampler for the training dataset
    train_sampler = DistributedSampler(train_dataset) if ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp else None

    # DataLoaders with the samplers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=4, pin_memory=True, shuffle=(train_sampler is None))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=4, pin_memory=True, shuffle=(val_sampler is None))

    return train_loader, val_loader

def get_cifar10(batch_size, strong_transform=False):
    """
    Returns train and validation data loaders for CIFAR-10 dataset.

    Args:
        batch_size (int): Number of samples per batch to load.

    Returns:
        tuple: A tuple of train and validation data loaders.
    """
    # Data preparation
    
    if strong_transform == 'v1':
        print("Using strong transform v1")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, shear=15, scale=(0.8,1.2)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.201])]
            )
    elif strong_transform == 'v2':
        print("Using strong transform v2")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.201])]
            )
    elif strong_transform == 'v3':
        print("Using strong transform v2")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.201])]
            )
    else:
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.201])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.201])
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def get_dataset(dataset_name, batch_size, strong_transform=False):
    print("Using dataset: " + dataset_name + " with batch size: " + str(batch_size) + " and strong transform: " + str(strong_transform))
    if dataset_name == 'cifar10':
        train_loader, val_loader = get_cifar10(batch_size, strong_transform)
    elif dataset_name == 'mnist':
        train_loader, val_loader =  get_mnist(batch_size, strong_transform)
    elif dataset_name == 'fashionmnist':
        # print(get_fashion_mnist)
        train_loader, val_loader = get_fashion_mnist(batch_size, strong_transform)
    return train_loader, val_loader

        
def get_fashion_mnist(batch_size, strong_transform=False):
    if strong_transform:
        print("Using strong transform")
        train_transform = transforms.Compose([transforms.RandomCrop(28, padding=4), 
                                        transforms.RandomHorizontalFlip(), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.5,), (0.5,))])
        
        val_transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.5,), (0.5,))])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), 
                                              transforms.Normalize((0.5,), (0.5,))])
        val_transform = train_transform
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=train_transform, download=True)
    val_dataset = datasets.FashionMNIST(root='./data', train=False, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def get_mnist(batch_size, strong_transform=False):
    # Data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
    

def get_cifar100(batch_size, strong_transform=False):
    """
    Returns train and validation data loaders for CIFAR-10 dataset.

    Args:
        batch_size (int): Number of samples per batch to load.

    Returns:
        tuple: A tuple of train and validation data loaders.
    """
    # Data preparation
    
   
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.4865, 0.4409],
                            std=[0.2673, 0.2564, 0.2761])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.4865, 0.4409],
                            std=[0.2673, 0.2564, 0.2761])
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True)
    val_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
