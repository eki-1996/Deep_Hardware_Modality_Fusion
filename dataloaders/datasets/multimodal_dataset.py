from __future__ import print_function, division
import os
from collections import OrderedDict
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms_multimodal as tr

class MultimodalDatasetSegmentation(Dataset):
    NUM_CLASSES = 20

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('multimodal_dataset'),
                 split='train',
                 ):
        """
        :param base_dir: path to KITTI dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self.modality = ['rgb', 'aolp', 'dolp', 'nir', 'image_000', 'image_045', 'image_090', 'image_135']
        assert args.dataset_modality.split(',') == self.modality
        for modality in args.use_modality.split(','):
            assert modality in self.modality
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'polL_color')
        self._cat_dir = os.path.join(self._base_dir, 'GT')
        # self._mask_dir = os.path.join(self._base_dir, 'SSmask')
        self._mask_dir = os.path.join(self._base_dir, 'SSGT4MS')
        self._aolp_sin_dir = os.path.join(self._base_dir, 'polL_aolp_sin')
        self._aolp_cos_dir = os.path.join(self._base_dir, 'polL_aolp_cos')
        self._dolp_dir = os.path.join(self._base_dir, 'polL_dolp')
        self._nir_dir = os.path.join(self._base_dir, 'NIR_warped')
        self._nir_mask_dir = os.path.join(self._base_dir, 'NIR_warped_mask')
        self._image_000_dir = os.path.join(self._base_dir, 'pol_I000')
        self._image_045_dir = os.path.join(self._base_dir, 'pol_I045')
        self._image_090_dir = os.path.join(self._base_dir, 'pol_I090')
        self._image_135_dir = os.path.join(self._base_dir, 'pol_I135')
        self._left_offset = 192


        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, 'list_folder')

        self.im_ids     = []
        self.images     = []
        self.aolp_sins  = []
        self.aolp_coss  = []
        self.dolps      = []
        self.nirs       = []
        self.nir_masks  = []
        self.categories = []
        self.mask = []
        self.images_000 = []
        self.images_045 = []
        self.images_090 = []
        self.images_135 = []


        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image    = os.path.join(self._image_dir    , line + ".png")
                _cat      = os.path.join(self._cat_dir      , line + ".png")
                _aolp_sin = os.path.join(self._aolp_sin_dir , line + ".npy")
                _aolp_cos = os.path.join(self._aolp_cos_dir , line + ".npy")
                _dolp     = os.path.join(self._dolp_dir     , line + ".npy")
                _nir      = os.path.join(self._nir_dir      , line + ".png")
                _nir_mask = os.path.join(self._nir_mask_dir , line + ".png")
                _mask=os.path.join(self._mask_dir      , line + ".png")
                _image_000 = os.path.join(self._image_000_dir    , line + ".png")
                _image_045 = os.path.join(self._image_045_dir    , line + ".png")
                _image_090 = os.path.join(self._image_090_dir    , line + ".png")
                _image_135 = os.path.join(self._image_135_dir    , line + ".png")
                
                if not os.path.isfile(_image):
                    continue
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.mask.append(_mask)
                self.aolp_sins.append(_aolp_sin)
                self.aolp_coss.append(_aolp_cos)
                self.dolps.append(_dolp)
                self.nirs.append(_nir)
                self.nir_masks.append(_nir_mask)
                self.images_000.append(_image_000)
                self.images_045.append(_image_045)
                self.images_090.append(_image_090)
                self.images_135.append(_image_135)
                
        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

        self.img_h = 1024
        self.img_w = 1224
        
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target, _aolp, _dolp, _nir, _nir_mask, mask, _img_000, _img_045, _img_090, _img_135 = self._make_img_gt_point_pair(index)
        sample = OrderedDict([('rgb', _img), ('label', _target), ('aolp', _aolp), ('dolp', _dolp), ('nir', _nir), ('nir_mask', _nir_mask), ('mask', mask), 
                  ('image_000', _img_000), ('image_045', _img_045), ('image_090', _img_090), ('image_135', _img_135)])

        for split in self.split:
            if split == "train":
                ret = self.transform_tr(sample)
            
            elif split == 'val':
                ret = self.transform_val(sample)
            elif split == 'test':
                ret = self.transform_val(sample)

        for modality in self.modality:
            if not modality in self.args.use_modality.split(','):
                del ret[modality]
        return ret


    def _make_img_gt_point_pair(self, index):
        _img = cv2.imread(self.images[index],-1)[:,:,::-1]
        _img = _img.astype(np.float32)/65535 if _img.dtype==np.uint16 else _img.astype(np.float32)/255
        _target = cv2.imread(self.categories[index],-1)
        if len(_target.shape) < 3: _target = np.expand_dims(_target, axis=2)
        #_target = np.load(self.categories[index])
        _mask = cv2.imread(self.mask[index],-1)
        if len(_mask.shape) < 3: _mask = np.expand_dims(_mask, axis=2)
        _aolp_sin = np.load(self.aolp_sins[index])
        _aolp_cos = np.load(self.aolp_coss[index])
        _aolp = np.stack([_aolp_sin, _aolp_cos], axis=2) # H x W x 2
        _dolp = np.load(self.dolps[index])
        if len(_dolp.shape) < 3: _dolp = np.expand_dims(_dolp, axis=2)
        _nir  = cv2.imread(self.nirs[index],-1)
        _nir = _nir.astype(np.float32)/65535 if _nir.dtype==np.uint16 else _nir.astype(np.float32)/255
        if len(_nir.shape) < 3: _nir = np.expand_dims(_nir, axis=2)
        _nir_mask = cv2.imread(self.nir_masks[index],0)
        if len(_nir_mask.shape) < 3: _nir_mask = np.expand_dims(_nir_mask, axis=2)
        _img_000 = cv2.imread(self.images_000[index],-1)
        _img_000 = _img_000.astype(np.float32)/65535 if _img_000.dtype==np.uint16 else _img_000.astype(np.float32)/255
        if len(_img_000.shape) < 3: _img_000 = np.expand_dims(_img_000, axis=2)
        _img_045 = cv2.imread(self.images_045[index],-1)
        _img_045 = _img_045.astype(np.float32)/65535 if _img_045.dtype==np.uint16 else _img_045.astype(np.float32)/255
        if len(_img_045.shape) < 3: _img_045 = np.expand_dims(_img_045, axis=2)
        _img_090 = cv2.imread(self.images_090[index],-1)
        _img_090 = _img_090.astype(np.float32)/65535 if _img_090.dtype==np.uint16 else _img_090.astype(np.float32)/255
        if len(_img_090.shape) < 3: _img_090 = np.expand_dims(_img_090, axis=2)
        _img_135 = cv2.imread(self.images_135[index],-1)
        _img_135 = _img_135.astype(np.float32)/65535 if _img_135.dtype==np.uint16 else _img_135.astype(np.float32)/255
        if len(_img_135.shape) < 3: _img_135 = np.expand_dims(_img_135, axis=2)
        return _img[:,self._left_offset:], _target[:,self._left_offset:], \
               _aolp[:,self._left_offset:], _dolp[:,self._left_offset:], \
               _nir[:,self._left_offset:], _nir_mask[:,self._left_offset:],_mask[:,self._left_offset:], \
               _img_000[:,self._left_offset:], _img_045[:,self._left_offset:], \
               _img_090[:,self._left_offset:], _img_135[:,self._left_offset:]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size,fill=255),
            tr.RandomGaussianBlur(),
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=1024),
            tr.FixScaleCrop(self.args.crop_size),
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'KITTI_material_dataset(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = MultimodalDatasetSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
    # plt.savefig('./out.png')


