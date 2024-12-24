class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'multimodal_dataset':
            return './datasets/multimodal_dataset/'  # folder that contains multimodal_dataset.
        elif dataset == 'rgb_thermal_dataset':
            return './datasets/rgb_thermal_dataset/'  # folder that contains rgb_thermal_dataset.
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
