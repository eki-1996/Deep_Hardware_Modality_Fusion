import argparse
import os
import numpy as np
from tqdm import tqdm
import random
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab_multi import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

from utils.gif_saver import save_gif

class TrainerMultimodal(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
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
                        fusion_kernal_size=args.fusion_kernal_size,
                        fused_out_dim=args.fused_out_dim,
                        )

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr*10},]
        
        # Define Optimizer
        optimizer = torch.optim.SGD(train_params,momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        # optimizer = torch.optim.SGD(train_params,momentum=args.momentum, nesterov=args.nesterov)

        # Define Criterion
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
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            #self.mask_model = torch.nn.DataParallel(self.mask_model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
    
    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        scaler = torch.cuda.amp.GradScaler()
        for i, sample in enumerate(tbar):
            if self.args.cuda:
                for key, item in sample.items():
                    item = item.cuda()
                    sample[key] = item
            if self.args.dataset == 'multimodal_dataset' and 'nir' in self.args.use_modality:
                nir_mask = sample.pop('nir_mask').squeeze(1)
            else: nir_mask = None
            label = sample.pop('label').squeeze(1)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()    
            
            with torch.cuda.amp.autocast():
                _, output = self.model(sample)
                loss = self.criterion(output, label, nir_mask)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
                
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('Loss: %.3f' % train_loss)

        # save checkpoint every epoch
        # if self.args.no_val:
        #     is_best = False
        #     self.saver.save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': self.model.module.state_dict(),
        #         'optimizer': self.optimizer.state_dict(),
        #         'best_pred': self.best_pred,
        #     }, is_best)

        # save the last epoch.
        if epoch == self.args.epochs - 1:
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best, filename='lastcheckpoint.pth.tar')


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        scaler = torch.cuda.amp.GradScaler()
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
                    _, output = self.model(sample)
                loss = self.criterion(output, label, nir_mask)
            test_loss += loss.item()
            tbar.set_description('Val loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            label_ = label.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(label_, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        
        print('Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            if True:
                self.test(epoch)
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def test(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        scaler = torch.cuda.amp.GradScaler()
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
                    fued_img, output = self.model(sample)
                loss = self.criterion(output, label, nir_mask)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            label_ = label.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(label_, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('test/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('test/mIoU', mIoU, epoch)
        self.writer.add_scalar('test/Acc', Acc, epoch)
        self.writer.add_scalar('test/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('test/fwIoU', FWIoU, epoch)
        
        print('Test:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        with open(os.path.dirname(os.path.abspath(__file__)) + '/'+ self.saver.experiment_dir + '/best_test.txt', 'a') as f:
            f.writelines(f"{mIoU}\n")


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
    parser.add_argument('--fusion-kernal-size', type=int, default=8, metavar='S',
                        help='hardware modality fusion kernal size (default: 8)')
    parser.add_argument('--fused-out-dim', type=int, default=1, metavar='S',
                        help='hardware mdoality fusion output channel (default: 1)')

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

    # default settings for epochs, batch_size and lr
    args.epochs = 500

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

    trainer = TrainerMultimodal(args)
    
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()
