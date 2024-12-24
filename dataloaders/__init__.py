from dataloaders.datasets import rgb_thermal_dataset, multimodal_dataset
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'multimodal_dataset':
        train_set = multimodal_dataset.MultimodalDatasetSegmentation(args, split='train')
        val_set = multimodal_dataset.MultimodalDatasetSegmentation(args, split='val')
        test_set = multimodal_dataset.MultimodalDatasetSegmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'rgb_thermal_dataset':
        train_set = rgb_thermal_dataset.MF_dataset(args=args, split='train')
        val_set  = rgb_thermal_dataset.MF_dataset(args=args, split='val')
        test_set = rgb_thermal_dataset.MF_dataset(args=args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError