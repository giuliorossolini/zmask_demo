#-----------------------------------------------------------------
# Neurons Distribution Adversarial Defense (NeuDAD)

# Author: Giulio Rossolini
# Scuola Superiore Sant'Anna Pisa
#-----------------------------------------------------------------
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict
from ptsemseg.utils import get_model_state
import torch
import os
import numpy as np
import patch_utils as patch_utils
import secrets
import zmask_utils 
from torch import nn
import torch.nn.functional as F


def load_patch_into_model(model, patch, train_loader):
    resume_path = patch
    #print("Resuming optimization from %s" % resume_path)
    seed_patch = patch_utils.get_patch_from_img(resume_path, set_loader=train_loader)
    patch_utils.init_model_patch(model = model, mode = "test", seed_patch = seed_patch)
    return model

def get_patch_from_set(patch_set):
    return secrets.choice(patch_set)


#----------------------------------------------------------------------------
#                               Loss Functions
#----------------------------------------------------------------------------

loss_function = nn.BCELoss()


def binary_cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ct, ht, wt = target.size()
    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        target = F.interpolate(target, size=(h, w), mode="bilinear", align_corners=True)
    input = input.view(-1)
    target = target.view(-1)
    loss = loss_function(input, target)
    return loss



def binary_cross_entropy(input, target):
    input = input.view(-1)
    target = target.view(-1)
    loss = loss_function(input, target)
    return loss



#-----------------------------------------------------------------------------
#                           Adversarial Detector
# Input: heatmap
# Output: mask and flag
#-----------------------------------------------------------------------------
class MaskBlock(nn.Module):
    def __init__(self, output_features=1):
        super(MaskBlock, self).__init__()
        self.bias = nn.Parameter(torch.zeros(output_features), requires_grad=True)
        self.weight = nn.Parameter(torch.zeros(output_features), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.conv_block_2A = nn.Conv2d(in_channels = 1, 
                out_channels = 1, 
                kernel_size=1, 
                stride = 1, 
                padding = 0, 
                bias=True)
        self.conv_block_2B = nn.Conv2d(in_channels = 1, 
                out_channels = 1, 
                kernel_size=1, 
                stride = 1, 
                padding = 0, 
                bias=True)
        self.conv_block_1A = nn.Conv2d(in_channels = 1, 
                out_channels = 1, 
                kernel_size=1, 
                stride = 1, 
                padding = 0, 
                bias=True)
        self.conv_block_head= nn.Conv2d(in_channels = 1, 
                out_channels = 1, 
                kernel_size=1, 
                stride = 1, 
                padding = 0, 
                bias=True)
        
        self.init_weights()


    def init_weights(self):
        torch.nn.init.uniform_((self.conv_block_2A.weight), a=0.0, b=1.0)
        torch.nn.init.uniform_((self.conv_block_2A.bias), a=0.0, b=1.0)
        torch.nn.init.uniform_((self.conv_block_2B.weight), a=0.0, b=1.0)
        torch.nn.init.uniform_((self.conv_block_2B.bias), a=0.0, b=1.0)
        torch.nn.init.uniform_((self.conv_block_1A.weight), a=0.0, b=1.0)
        torch.nn.init.uniform_((self.conv_block_1A.bias), a=0.0, b=1.0)
        torch.nn.init.uniform_((self.conv_block_head.weight), a=0.0, b=1.0)
        torch.nn.init.uniform_((self.conv_block_head.bias), a=0.0, b=1.0)
        torch.nn.init.uniform_((self.weight), a=0.0, b=1.0)
        torch.nn.init.uniform_((self.bias), a=0.0, b=1.0)



    def forward(self, f1, f2):
        f1 = self.tanh(self.conv_block_2A(f1))
        f1 = self.conv_block_1A(f1)
        f1 = self.sigmoid(f1)
        f = f2 * f1

        f = self.tanh(self.conv_block_2B(f))
        out1 = self.conv_block_head(f)
        out1 = self.sigmoid(out1)
        return out1


class DefenseAndDetectorNN(nn.Module):
    def __init__(self):
        super(DefenseAndDetectorNN, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.MaskBlock = MaskBlock(output_features = 1)


    def forward_mask(self, x1, x2):
        mask = self.MaskBlock(x1,x2)
        return mask

    
    def forward(self, f1, f2):
        mask = self.forward_mask(f1, f2)
        mask = F.interpolate(mask, size=f1.shape[2:],mode='bilinear', align_corners=True)
        f = (f1*f2) * mask
        out = (f.sum(axis=(1,2,3)).unsqueeze(1))/(1 + mask.sum(axis=(1,2,3)).unsqueeze(1))
        return out, mask




#------------------------------------------------------------------------
#------------------------------------------------------------------------

#------------------------------------------------------------------------
#------------------------------------------------------------------------
class ZMASK(object):
    def __init__(self, spatial_layer, spatial_kernels, context_layer, context_kernels, device, size_fig=(150,300), num_planes=1):

        self.spatial_layer = spatial_layer
        self.spatial_kernels = spatial_kernels
        self.context_layer = context_layer
        self.context_kernels = context_kernels
        self.size_fig = size_fig
        self.num_planes = num_planes
        self.device = device
        self.mean_layers = {}
        self.std_layers = {}
        self.DefenseBlock = DefenseAndDetectorNN()
        self.thr_spatial = None
        self.thr_context = None


    #--------------------------------------------------------
    # compute mean and std 
    #--------------------------------------------------------  
    def init_mean_and_std(self):
        self.mean_layers = {}
        self.std_layers = {}
        return self.mean_layers, self.std_layers
        
    
    def compute_mean_and_std(self,model, loader):
        layers = dict(list(self.spatial_layer.items()) + list(self.context_layer.items()))
        for index in range(len(loader)):
            image = loader[index][0].unsqueeze(0).to(self.device)
            acts = zmask_utils.get_activations(model.to(self.device), layers, image)
            
            for layer in layers:
                act = acts['activations'][layer][0]
                
                act = torch.tensor(act).to('cuda').unsqueeze(0)
                act = F.interpolate(act, size=self.size_fig)                
                act = act.squeeze(0)
                act = act.view(act.shape[0], -1)
                act = act.view(act.shape[0], self.num_planes, -1)
                if self.std_layers.get(layer, None) is None:
                    self.std_layers[layer] = act.std(axis=(2), keepdims=True).cpu().numpy()
                    self.mean_layers[layer] = act.mean( axis=(2), keepdims=True).cpu().numpy()
                else:
                    self.std_layers[layer] += act.std(axis=(2), keepdims=True).cpu().numpy()
                    self.mean_layers[layer] +=act.mean(axis=(2), keepdims=True).cpu().numpy()

        for layer in layers:
            self.std_layers[layer] /= len(loader)
            self.mean_layers[layer] /= len(loader)
        
        return self.std_layers, self.mean_layers


    #------------------------------------------------------------
    #------------------------------------------------------------
    # get single layer heatmap
    def get_single_layer_heatmap(self, act, layer_name, kernels):
        tot_heatmap, heatmap = None, None
        for kernel in kernels:
            # TODO: migliorare per svolgere una sola volta avgpool
            heatmap = self.get_heatmap(act, layer_name, kernel)
            if tot_heatmap is None:
                tot_heatmap = heatmap
            else:
                tot_heatmap /= (1+tot_heatmap.view((tot_heatmap.shape[0], 
                    tot_heatmap.shape[1], -1)).max(dim=-1)[0].unsqueeze(2).unsqueeze(3))
                tot_heatmap *= heatmap

        return tot_heatmap


    #------------------------------------------------------------
    #------------------------------------------------------------
    # get aggregated heatmap 
    def get_aggregated_heatmap(self,act, layers, kernels):
        final_heatmap = None
        for key in list(layers.keys()):
            if final_heatmap is None: 
                final_heatmap = self.get_single_layer_heatmap(act,key, kernels)
            else:
                final_heatmap = torch.max(final_heatmap, self.get_single_layer_heatmap(act, key, kernels))
            
        return final_heatmap

    #------------------------------------------------------------
    #------------------------------------------------------------
    # Return internal heatmap from context and spatial (if double is true)
    def run_analysis(self, act, double_analysis=True):
        final_context_heatmap = None
        final_spatial_heatmap = None
        if double_analysis is True:
            final_context_heatmap = self.get_aggregated_heatmap(act, self.context_layer, self.context_kernels)
        
        final_spatial_heatmap = self.get_aggregated_heatmap(act, self.spatial_layer, self.spatial_kernels)
    
        return final_context_heatmap, final_spatial_heatmap


    #-----------------------------------------------------------
    # get the heat map from a given layer
    def get_heatmap(self, act, layer, kernel):
        act = act['activations'][layer][0]
        heat = torch.tensor(act).to('cuda')
        if(len(heat.shape) == 3):
            heat = heat.unsqueeze(0)
        ad = nn.AvgPool2d(kernel, stride=1)
        # normalize activations
        heat = F.interpolate(heat, size=self.size_fig)
        b,c,h,w = heat.shape[0],heat.shape[1],heat.shape[2],heat.shape[3]
        #print(heat.shape)
        heat = heat.view(b,c, -1)
        heat = heat.view(b,c, self.num_planes, -1)
        features_std = torch.tensor(self.std_layers[layer]).to(self.device).unsqueeze(0)
        #features_std[features_std<1e-3] = 1
        faetures_mean = torch.tensor(self.mean_layers[layer]).to(self.device).unsqueeze(0)
        heat = (heat - faetures_mean)/(features_std + 1e-5)
        heat = heat.view(b,c, h, w)
        # pooling block
        heat = torch.tensor(heat).to('cuda')
        heat = F.interpolate(heat, size=self.size_fig)
        heat = ad(heat).abs()
        heat = F.interpolate(heat, size=self.size_fig)
        heat = heat.mean(axis=1, keepdims=True)
        return heat 



    def forward_defense(self, x, model, thr_mask = False, use_thr = False):
        act_spatial, act_context, act_spatial_name, act_context_name = model.forward_and_get_activations(x)
        #final_context_heatmap, final_spatial_heatmap = self.QUICK_run_analysis(act_spatial, act_context, act_spatial_name, act_context_name)
        final_context_heatmap, final_spatial_heatmap = self.run_analysis(act_spatial, act_context, act_spatial_name, act_context_name)
        flag, mask = self.DefenseBlock(final_context_heatmap, final_spatial_heatmap)

        if use_thr:
            is_attack = (flag > self.flag_thr)
            is_attack = is_attack.unsqueeze(1).unsqueeze(2)
            mask *= (is_attack)
    
        mask = 1-mask
        mask = mask.repeat(1,3,1,1)
        mask = F.interpolate(mask, size=x.shape[2:])  

        if thr_mask:
            # TODO - change with a hvside function
            mask[mask > 0.5] = 1.0
            mask[mask < 0.5] = 0.0
   
        data = x * mask
        outputs = model(data) 
        return outputs, mask, flag



def test_ZMASK(cfg, 
                loader, 
                n_classes, 
                ZMASK, 
                patch = None, 
                patch_params = None,
                output_file = None, 
                use_transformations=False):

    running_metrics = runningScore(n_classes)

    ex_clear_image, ex_adv_image      =  None, None
    ex_clear_out, ex_adv_out          =  None, None

    device = torch.device("cuda")
    torch.cuda.set_device(cfg["device"]["gpu"])

    # Setup Model and patch
    model_file_name = os.path.split(cfg["model"]["path"])[1]
    model_name = model_file_name[: model_file_name.find("_")]
    print(model_name)

    model_dict = {"arch": cfg["model"]["arch"]}
    model = get_model(model_dict, n_classes, version=cfg["data"]["dataset"])
    state = torch.load(cfg["model"]["path"], map_location = 'cpu')
    state = get_model_state(state, model_name)
    model.load_state_dict(state)    
    patch_utils.init_model_patch(model = model, mode = "test", seed_patch = patch)

    ex_index = 2
    iteration = 0

    model = model.eval()
    model = model.to(device)

    for i, (images, labels) in enumerate(loader):

        with torch.no_grad():

            images = images.to(device)
            if isinstance(labels, list):
                labels, extrinsic, intrinsic = labels
                extrinsic, intrinsic = extrinsic.to(device), intrinsic.to(device)

            #-------------------------------------------------------------------
            # Compute an example image for visualiation
            #-------------------------------------------------------------------
            if ex_clear_image is None:
                ex_clear_image = images[ex_index].clone().reshape(1,images.shape[1], images.shape[2], images.shape[3])
                ex_clear_out = model(ex_clear_image)

                ex_adv_image = patch_utils.add_patch_to_batch(
                    ex_clear_image.clone(), 
                    patch = model.patch, 
                    device = device, 
                    use_transformations=use_transformations, 
                    patch_params=patch_params,
                    int_filtering=True)[0]

                # interporale perturbed_images at the masknet size
                ex_adv_out,_,_= ZMASK.forward_defense(ex_adv_image, model, thr_mask=True)

            #-------------------------------------------------------------------

            perturbed_images, label_mask = patch_utils.add_patch_to_batch(
                    images, 
                    patch = model.patch, 
                    device = device, 
                    use_transformations=use_transformations, 
                    patch_params=patch_params,
                    int_filtering=True)

            outputs,mask,_ = ZMASK.forward_defense(perturbed_images, model, thr_mask=True)

            pred = outputs.data.max(1)[1].cpu().numpy()

            gt = labels.numpy()
            gt_woPatch = patch_utils.remove_mask(gt, label_mask.detach())
            running_metrics.update(gt_woPatch, pred)

        

    score, class_iou = running_metrics.get_scores()


    return score, class_iou, ex_clear_image, ex_adv_image, ex_clear_out, ex_adv_out  