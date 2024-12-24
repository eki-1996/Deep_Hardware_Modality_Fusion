import argparse
import os
import sys
from matplotlib import animation
import numpy as np
from tqdm import tqdm
import random
# import matplotlib  # <--ここを追加
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

from mypath import Path
from dataloaders import make_data_loader
from dataloaders.utils import get_three_channels_visulized_image
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab_multi import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

        
class TesterMultimodal(object):
    def __init__(self, args):
        self.args = args

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(f'{os.path.dirname(args.pth_path)}/test_results')
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLabMultiInput(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        dataset=args.dataset,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,
                        fusion_input_dim=[int(x) for x in args.input_channels_list.split(',')],
                        ratio=args.ratio,
                        pretrained=args.use_pretrained_resnet,
                        use_hardware_modality_fusion=args.use_hardware_modality_fusion,
                        fusion_kernel_size=args.fusion_kernel_size,
                        fused_out_dim=args.fused_out_dim,
                        )
        
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        # self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda, ignore_index=0).build_loss(mode=args.loss_type)
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model = model

        # Load model parameters
        checkpoint = torch.load(args.pth_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

    def test(self, epoch=0):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        scaler = torch.cuda.amp.GradScaler()
        image_all = None
        label_all = None
        output_all = None
        fuesd_img_all = None
        for i, sample in enumerate(tbar):
            if self.args.cuda:
                for key, item in sample.items():
                    item = item.cuda()
                    sample[key] = item
            if self.args.dataset == 'multimodal_dataset' and 'nir' in self.args.use_modality:
                nir_mask = sample.pop('nir_mask').squeeze(1)
            else: nir_mask = None
            label = sample.pop('label').squeeze(1)     

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    fused_img, output = self.model(sample)
                loss = self.criterion(output, label, nir_mask)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            
            pred = output.data.cpu().numpy()
            converted_3c_img = get_three_channels_visulized_image(sample)
            label_ = label.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            if image_all is None:
                image_all  =  converted_3c_img.cpu().clone()
                label_all = label.cpu().clone()
                output_all = output.cpu().clone()
                fuesd_img_all = fused_img.cpu().clone()
            else:
                image_all  = torch.cat((image_all, converted_3c_img.cpu().clone()),dim=0)
                label_all = torch.cat((label_all,label.cpu().clone()),dim=0)
                output_all = torch.cat((output_all,output.cpu().clone()),dim=0)
                fuesd_img_all = torch.cat((fuesd_img_all,fused_img.cpu().clone()),dim=0)
            # Add batch sample into evaluator
            self.evaluator.add_batch(label_, pred)
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        confusion_matrix = self.evaluator.confusion_matrix
        np.save(os.path.join(self.summary.directory, 'confusion_matrix.npy'),confusion_matrix)

        self.writer.add_scalar('visualize/mIoU', mIoU, epoch)
        self.writer.add_scalar('visualize/Acc', Acc, epoch)
        self.writer.add_scalar('visualize/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('visualize/fwIoU', FWIoU, epoch)
        self.summary.visualize_test_image(self.writer, self.args.dataset, image_all, label_all, output_all, fuesd_img_all, 0)
        
        print('Test:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--input-channels-list', type=str, default='3',
                        help='network input channels (default: 3)')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet', 'resnet_adv', 'xception_adv','resnet_condconv'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='multimodal_dataset', 
                        help='dataset name (default: multimodal_dataset)')
    parser.add_argument('--dataset-modality', type=str, default='rgb',
                        help='dataset modality (default: rgb)')
    parser.add_argument('--use-modality', type=str, default='rgb',
                        help='dataset use modality (default: rgb)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=True,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal', 'original','bce'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--ratio', type=float, default=None, metavar='N',
                        help='number of ratio in RGFSConv (default: 1)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # propagation and positional encoding option
    parser.add_argument('--propagation', type=int, default=0,
                        help='image propagation length (default: 0)')
    parser.add_argument('--positional-encoding', action='store_true', default=False,
                        help='use positional encoding')
    parser.add_argument('--use-pretrained-resnet', action='store_true', default=False,
                        help='use pretrained resnet101')
    parser.add_argument('--use-hardware-modality-fusion', action='store_true', default=False,
                        help='use hardware modality fusion to fuse modalities')
    parser.add_argument('--fusion-kernel-size', type=int, default=8, metavar='S',
                        help='hardware modality fusion kernel size (default: 8)')
    parser.add_argument('--fused-out-dim', type=int, default=1, metavar='S',
                        help='hardware mdoality fusion output channel (default: 1)')
    
    parser.add_argument('--pth-path', type=str, default=None,
                        help='set the pth file path')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False


    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    args.lr = 0.007

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    # input('Check arguments! Press Enter...')
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    tester = TesterMultimodal(args)
    tester.test()
    tester.writer.close()