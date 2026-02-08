import math
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset, default_collate
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.distributions import gamma
from typing import Tuple
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CIFARDataLoader:  # FIXED: Renamed from DataLoader to avoid conflict
    """Unified dataset loader for HSGSP project"""
    
    def __init__(self, config):
        self.config = config
        self.root = getattr(config, 'data_root', './data')
    
    def _mixup_collate(self, num_classes: int):
        use_mixup = bool(getattr(self.config, "use_mixup", False))
        mixup_alpha = float(getattr(self.config, "mixup_alpha", 0.0))
        mixup_prob = float(getattr(self.config, "mixup_prob", 0.0))
        
        if not use_mixup or mixup_alpha <= 0.0 or mixup_prob <= 0.0:
            return self._one_hot_collate(num_classes)
        
        def collate(batch):
            images, labels = default_collate(batch)
            labels = F.one_hot(labels, num_classes=num_classes).float()
            
            if torch.rand(1).item() < mixup_prob:
                alpha_tensor = torch.tensor(mixup_alpha)
                indices = torch.randperm(images.size(0))
                shuffled_images = images[indices]
                shuffled_labels = labels[indices]
                
                gamma1 = gamma.Gamma(alpha_tensor, torch.tensor(1.0)).sample((1,))
                gamma2 = gamma.Gamma(alpha_tensor, torch.tensor(1.0)).sample((1,))
                lam = gamma1 / (gamma1 + gamma2)
                lam = lam.to(images.dtype).to(images.device)
                
                images = lam * images + (1.0 - lam) * shuffled_images
                labels = lam * labels + (1.0 - lam) * shuffled_labels
            
            return images, labels
        
        return collate
    
    def _one_hot_collate(self, num_classes: int):
        def collate(batch):
            images, labels = default_collate(batch)
            labels = F.one_hot(labels, num_classes=num_classes).float()
            return images, labels
        
        return collate
    
    def load_cifar10(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Load and preprocess CIFAR-10 dataset.
        
        Returns:
            train_dl: augmented training dataloader (with stochastic transforms)
            val_dl: validation dataloader without augmentation
            test_dl: test dataloader without augmentation
            train_eval_dl: clean view of the training set (no augmentation) for evaluation
        """
        base_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        augment_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.2),
            transforms.RandomErasing(p=0.5, scale=(0.0625, 0.25), ratio=(1.0, 1.0), value=0.0),
        ]) if self.config.data_augmentation else base_transform
        
        # Load full train dataset with no transform to get consistent split
        full_train = datasets.CIFAR10(root=self.root, train=True, download=True, transform=None)
        
        # Split with seed
        generator = torch.Generator().manual_seed(SEED)
        val_size = int(len(full_train) * self.config.validation_split)
        train_size = len(full_train) - val_size
        train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=generator)
        
        train_indices = train_subset.indices
        val_indices = val_subset.indices
        
        # Create datasets with appropriate transforms using the same root
        train_dataset = datasets.CIFAR10(root=self.root, train=True, download=False, transform=augment_transform)
        plain_dataset = datasets.CIFAR10(root=self.root, train=True, download=False, transform=base_transform)
        
        train_ds = Subset(train_dataset, train_indices)
        val_ds = Subset(plain_dataset, val_indices)
        train_eval_ds = Subset(plain_dataset, train_indices)
        test_ds = datasets.CIFAR10(root=self.root, train=False, download=True, transform=base_transform)
        
        num_classes = self.config.num_classes_cifar10
        num_workers = getattr(self.config, 'num_workers', 4)
        
        train_dl = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True,
                             collate_fn=self._mixup_collate(num_classes), num_workers=num_workers, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False,
                           collate_fn=self._one_hot_collate(num_classes), num_workers=num_workers, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False,
                            collate_fn=self._one_hot_collate(num_classes), num_workers=num_workers, pin_memory=True)
        train_eval_dl = DataLoader(train_eval_ds, batch_size=self.config.batch_size, shuffle=False,
                                   collate_fn=self._one_hot_collate(num_classes), num_workers=num_workers, pin_memory=True)
        
        return train_dl, val_dl, test_dl, train_eval_dl
    
    def load_cifar100(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Load and preprocess CIFAR-100 dataset.
        
        Returns:
            train_dl: augmented training dataloader
            val_dl: validation dataloader without augmentation
            test_dl: test dataloader without augmentation
            train_eval_dl: clean training dataloader for evaluation/monitoring
        """
        base_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        augment_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.2),
            transforms.RandomErasing(p=0.5, scale=(0.0625, 0.25), ratio=(1.0, 1.0), value=0.0),
        ]) if self.config.data_augmentation else base_transform
        
        # Load full train dataset with no transform to get consistent split
        full_train = datasets.CIFAR100(root=self.root, train=True, download=True, transform=None)
        
        # Split with seed
        generator = torch.Generator().manual_seed(SEED)
        val_size = int(len(full_train) * self.config.validation_split)
        train_size = len(full_train) - val_size
        train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=generator)
        
        train_indices = train_subset.indices
        val_indices = val_subset.indices
        
        # Create datasets with appropriate transforms using the same root
        train_dataset = datasets.CIFAR100(root=self.root, train=True, download=False, transform=augment_transform)
        plain_dataset = datasets.CIFAR100(root=self.root, train=True, download=False, transform=base_transform)
        
        train_ds = Subset(train_dataset, train_indices)
        val_ds = Subset(plain_dataset, val_indices)
        train_eval_ds = Subset(plain_dataset, train_indices)
        test_ds = datasets.CIFAR100(root=self.root, train=False, download=True, transform=base_transform)
        
        num_classes = self.config.num_classes_cifar100
        num_workers = getattr(self.config, 'num_workers', 4)
        
        train_dl = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True,
                             collate_fn=self._mixup_collate(num_classes), num_workers=num_workers, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False,
                           collate_fn=self._one_hot_collate(num_classes), num_workers=num_workers, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False,
                            collate_fn=self._one_hot_collate(num_classes), num_workers=num_workers, pin_memory=True)
        train_eval_dl = DataLoader(train_eval_ds, batch_size=self.config.batch_size, shuffle=False,
                                   collate_fn=self._one_hot_collate(num_classes), num_workers=num_workers, pin_memory=True)
        
        return train_dl, val_dl, test_dl, train_eval_dl
