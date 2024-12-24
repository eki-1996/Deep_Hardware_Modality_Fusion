import torch
import torch.nn as nn
import torch.nn.functional as F
# import pathlib
# import sys
# sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from torchvision import transforms

from modeling.hardware_modality_fusion_module import Exposuref

class DeepLabMultiInput(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, dataset='multimodal_dataset', num_classes=1,
                 sync_bn=True, freeze_bn=False, fusion_input_dim = [3], ratio=1, pretrained=False, 
                 use_hardware_modality_fusion=True, fusion_kernal_size=8, fused_out_dim=1):
        
        super(DeepLabMultiInput, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.fused_out_dim = fused_out_dim
        self.use_hardware_modality_fusion = use_hardware_modality_fusion
        self.dataset = dataset

        if use_hardware_modality_fusion:
            self.exp0 = Exposuref(in_channel=sum(fusion_input_dim), out_channel=fused_out_dim, kernal_size=fusion_kernal_size, binarize_type='full')
            self.transforms = transforms.Compose([transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            fusion_input_dim = [3]

        forward_encoder_modules = []
        for i, input_dim in enumerate(fusion_input_dim):
            setattr(self, f'backbone{i+1}', build_backbone(backbone, output_stride, BatchNorm, input_dim=input_dim, pretrained=pretrained))
            setattr(self, f'aspp{i+1}', build_aspp(backbone, output_stride, BatchNorm))
            forward_encoder_modules.extend([f'backbone{i+1}', f'aspp{i+1}'])

        self.decoder = build_decoder(num_classes, backbone, BatchNorm, ratio, input_heads=len(fusion_input_dim), use_hardware_modality_fusion=self.use_hardware_modality_fusion)

        self.forward_encoder_modules = forward_encoder_modules

        self.freeze_bn = freeze_bn

    def forward(self, input):
        if 'mask' in input.keys():
            mask = input.pop('mask')
        else: mask = None

        fused_modality = None
        if self.use_hardware_modality_fusion:
            input = torch.unsqueeze(torch.cat([x for x in input.values()],dim=1), dim=1)
            fused_modality = self.exp0(input)
            if self.fused_out_dim == 1:
                fused_modality = torch.cat([fused_modality,fused_modality,fused_modality], dim=1)
            elif self.fused_out_dim == 2:
                fused_modality_tmp = torch.unsqueeze(fused_modality.mean(dim=1), dim=1)
                fused_modality = torch.cat([fused_modality,fused_modality_tmp], dim=1)
            elif self.fused_out_dim == 3:
                fused_modality = fused_modality
            else:
                raise NotImplementedError("Your fused_out_dim does not implemented!")
            
            if not self.dataset == 'rgb_thermal_dataset':
                fused_modality = self.transforms(fused_modality)
        
            input = {'fused_modality': fused_modality}

        low_level_feats = []
        high_level_feats = []
        assert len(input.values()) == len(self.forward_encoder_modules)/2
        for i in range(int(len(self.forward_encoder_modules)/2)):
            high_level_feat_tmp, low_level_feat_tmp = getattr(self, self.forward_encoder_modules[i*2])(list(input.values())[i])
            high_level_feat_tmp = getattr(self, self.forward_encoder_modules[i*2 + 1])(high_level_feat_tmp)
            low_level_feats.append(low_level_feat_tmp)
            high_level_feats.append(high_level_feat_tmp)

        low_level_feat = torch.cat(low_level_feats,dim=1)
        high_level_feat = torch.cat(high_level_feats,dim=1) 

        output = self.decoder(high_level_feat, low_level_feat, mask)

        output = F.interpolate(output, size=list(input.values())[0].size()[2:], mode='bilinear', align_corners=True)

        return fused_modality, output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [getattr(self, x) for x in self.forward_encoder_modules if 'backbone' in x]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                #print(p)
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], Exposuref):
                        for p in m[1].parameters():
                            if p.requires_grad:
                               # print(p)
                                yield p

    def get_10x_lr_params(self):
        modules = [self.decoder]
        if self.use_hardware_modality_fusion:
            modules.append(self.exp0)
            modules.extend([getattr(self, x) for x in self.forward_encoder_modules if 'aspp' in x])
        else:
            modules.extend([getattr(self, x) for x in self.forward_encoder_modules if 'aspp' in x])
        for i in range(len(modules)):
            for m in modules[i].named_modules():

                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                
                                yield p
                else:
            
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1],nn.Linear) or isinstance(m[1], Exposuref):
                        for p in m[1].parameters():
                            #if m[0].split('.')[0]=='condconv':
                                #continue
                            if p.requires_grad:
                                #print(m[0])
                                yield p
                    if m[0]=='gamma':
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
    '''
    def get_100x_lr_params(self):
        modules = [self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():

                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                #print(m[0])
                                yield p
                else:
                    if m[0].split('.')[0]=='condconv':
                        for p in m[1].parameters():
                            if p.requires_grad:
                                #print(m[0])
                                yield p
        #for  m, parameters in modules[4].named_parameters():
            #for p in parameters:
                #if p.requires_grad:
                    #yield p
    '''

if __name__ == "__main__":
    model = DeepLabMultiInput(backbone='resnet_adv', output_stride=16, num_classes=10, fusion_input_dim=[3, 2, 1, 1], pretrained=True, use_hardware_modality_fusion=True)
    print(dir(model))
    model.eval()
    input = {'rgb': torch.rand(1, 3, 512, 512), 'aolp': torch.rand(1, 2, 512, 512), 'dolp': torch.rand(1, 1, 512, 512), 'nir': torch.rand(1, 1, 512, 512), 'mask': torch.rand(1, 1, 512, 512)}
    _, output = model(input)
    print(output.size())


