import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import platform
import enum
import copy
from bisect import bisect_left
import warnings

use_cuda = torch.cuda.is_available()

#=================================================================================
def get_activations(model, layers, image, label = None):
    activations = {}
    activations['activations'] = {}
    activations['targets'] = None

    for layer_name in layers:
        #print('## Fetching Activations from Layer {}'.format(layer_name))

        # Get activations for the data
        layer = layers[layer_name]
        activations['activations'][layer_name], targets = get_activations_from_layer(model, layer, image, label)

        # Get the targets of that data
        if targets is not None:
            if activations['targets'] is not None:
                np.testing.assert_array_equal(activations['targets'], targets)
            else:
                activations['targets'] = targets
    return activations


#=================================================================================
def get_activations_from_layer(model, layer, image, label):
    activations = []
    targets = []
    
    #------------------------------------------------------------
    # Define hook for fetching the activations
    def hook(module, input, output):
        check_list = isinstance(output, list) or type(output) is tuple        
        if(check_list):
            layers_activations_list = []
            for i in range(len(output)):
                layer_activations = output[i].squeeze().detach().cpu().numpy()
                if len(layer_activations.shape) >= 4:
                    layers_activations_list.append(layer_activations)#.reshape(layer_activations.shape[0], -1))
                #layers_activations_list.append(layer_activations)
            #layers_activations_list = np.stack(layers_activations_list)
            activations.append(layers_activations_list)
            return
        else:
            layer_activations = output.squeeze().detach().cpu().numpy()
            if len(layer_activations.shape) >= 4:
                layer_activations = layer_activations#.reshape(layer_activations.shape[0], -1)
            activations.append(layer_activations)
            #activations = np.concatenate(activations)
    #------------------------------------------------------------

    handle = layer.register_forward_hook(hook)
    
    # Fetch activations

    if use_cuda:
        batch = image.cuda()
    
    _ = model(batch)

    if len(batch) > 1:
        targets.append(batch[1].detach().cpu().numpy())
        
    # Remove hook
    handle.remove()

    # Return activations and targets
    #activations = np.concatenate(activations)
    
    if targets:
        targets = np.hstack(targets)
    else:
        None

    return activations, targets


#=================================================================================
def get_patch_mask(patch_size=(0,0), patch_pos=(None,None), image_size=(0,0)):
    if(patch_pos[0] is None or patch_pos[1] is None):
        patch_x = int((image_size[1]- patch_size[1])/2)
        patch_y = int((image_size[0]- patch_size[0])/2)
    else:
        patch_x, patch_y = patch_pos
    
    patch_mask = np.ones(image_size)
    patch_mask[patch_y:patch_y + patch_size[0], patch_x:patch_x + patch_size[1]] = 0
    return patch_mask    
#=================================================================================


#=================================================================================
from scipy import ndimage
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)
#=================================================================================


#=================================================================================
def get_spatial_areas(patch_size=(0,0), patch_pos=(None,None), 
                      image_size=(0,0), num_areas = 10):
    
    areas_array = [] 
    offset_areas_x = int((image_size[1]-patch_size[1])/num_areas)
    offset_areas_y = int((image_size[0]-patch_size[0])/num_areas)
    
    patch_size_x = patch_size[1]
    patch_size_y = patch_size[0]
    
    for i in range(num_areas+1):
    
        #TODO: extend this code also for patches not in the center of the image
        if(patch_pos[0] is None or patch_pos[1] is None):
            patch_x = int((image_size[1]- patch_size_x)/2)
            patch_y = int((image_size[0]- patch_size_y)/2)
        else:
            patch_x, patch_y = patch_pos

        patch_mask = np.zeros(image_size)
        if i == num_areas + 1:
            patch_mask[patch_y:patch_y + patch_size_y, patch_x:] = 1
        else:
            patch_mask[patch_y:patch_y + patch_size_y, patch_x:patch_x + patch_size_x] = 1
        
        
        for prev_mask in areas_array:
            patch_mask = patch_mask - prev_mask
            
        areas_array.append(patch_mask)
        patch_size_x += offset_areas_x
        patch_size_y += offset_areas_y
        
    return areas_array    
#=================================================================================

#=================================================================================
def get_borders(patch_size=(0,0), 
                patch_pos=(None,None), 
                image_size=(0,0), 
                num_areas = 10, 
                border_size = 2):
    
    areas_array = [] 
    offset_areas_x = int((image_size[1]-patch_size[1])/num_areas)
    offset_areas_y = int((image_size[0]-patch_size[0])/num_areas)
    
    size_b = border_size
    
    patch_size_x = patch_size[1]
    patch_size_y = patch_size[0]
    
    for i in range(num_areas+1):
    
        #TODO: extend this code also for patches not in the center of the image
        if(patch_pos[0] is None or patch_pos[1] is None):
            patch_x = int((image_size[1]- patch_size_x)/2)
            patch_y = int((image_size[0]- patch_size_y)/2)
        else:
            patch_x, patch_y = patch_pos

        patch_mask = np.zeros(image_size)
        c_patch_mask = np.zeros(image_size)
        patch_mask[patch_y  :patch_y + patch_size_y, patch_x:patch_x + patch_size_x] = 1
        c_patch_mask[patch_y+size_b:patch_y + patch_size_y-size_b, patch_x+size_b:patch_x + patch_size_x-size_b] = 1
        patch_mask -= c_patch_mask
        
        
        #for prev_mask in areas_array:
            #patch_mask = patch_mask + prev_mask
            
        areas_array.append(patch_mask)
        patch_size_x += offset_areas_x
        patch_size_y += offset_areas_y
        
    return areas_array    
#=================================================================================


'''
def incremental_mean(x, counter, S1):
    new_S1 = S1 + x
    return new_S1/counter


def incremental_std(x, counter, S1, S2):
    pow_x = torch.pow(x)
    new_S1 = S1 + x
    new_S2 = S2 + pow_x
    std = new_S2/counter - (new_S1/counter).pow()
    return std, new_S1, new_S2
'''

#=================================================================================
def get_mean_and_std_features(model, layers, dataloader):
    stats = {}
    stats['mean'] = {}
    stats['std'] = {}
    
    for layer_name in layers:
        print('## Fetching Activations from Layer {}'.format(layer_name))
        # Get activations for the data
        layer = layers[layer_name]
        stats['mean'][layer_name], stats['std'][layer_name] = get_mean_and_std_from_layer(model, layer, dataloader)
    return stats



#=================================================================================


def get_mean_and_std_from_layer(model, layer, dataloader):
    S1, S2, N = [0],[0],[0]
    def hook(module, input, output):
        nonlocal S1, S2, N
        check_list = isinstance(output, list) or type(output) is tuple
        
        if(check_list):
            for i in range(len(output)):
                if len(S1) < (i+1):
                    S1.append(0)
                    S2.append(0)
                    N.append(0)
            
                s1, s2, n = S1[i], S2[i], N[i]
                n += output[i].shape[0]
                layers_activations = output[i].squeeze().detach().cpu().numpy()
                if(len(layers_activations.shape) < 4):
                    channel_means = layers_activations.mean(axis=0)
                    channel_dev = layers_activations.std(axis=0)
                else:
                    channel_means = layers_activations.mean(axis=(0,2,3))
                    channel_dev = layers_activations.std(axis=(0,2,3))
                s1 += channel_means
                #s2 += channel_means**2
                s2 += channel_dev
                S1[i], S2[i], N[i] = s1, s2, n
        else:
            s1, s2, n = S1[0], S2[0], N[0]
            n += output.shape[0]
            layers_activations = output.squeeze().detach().cpu().numpy()
            if(len(layers_activations.shape) < 4):
                channel_means = layers_activations.mean(axis=0)
                channel_dev = layers_activations.std(axis=0)
            else:
                channel_means = layers_activations.mean(axis=(0,2,3))
                channel_dev = layers_activations.std(axis=(0,2,3))
            s1 += channel_means
            #s2 += channel_means**2
            s2 += channel_dev
            S1[0], S2[0], N[0] = s1, s2, n
     
    
    handle = layer.register_forward_hook(hook)
    
    # Fetch activations
    for i, batch in enumerate(dataloader):
        #print("Processing Batch{}".format(i))
        if use_cuda:
            images = batch[0].cuda()
        _ = model(images)
        
    # Remove hook
    handle.remove()
    
    mean, std = [], []
    for i in range(len(S1)):
        mean.append(S1[i]/N[i])
        #std.append(S1[i]/N[i] - (S2[i]/N[i])**2)
        std.append(S2[i]/np.sqrt(N[i]))
    
    return mean, std

#------------------------------------------------------
# 
#------------------------------------------------------

import sys 
sys.path.append('..')
import patch_utils
import cv2


def run_layers_robust_analysis(
    dataloader, 
    model, 
    patch, 
    layers, 
    patch_params,
    num_areas = 10,
    device = 'cuda',
    use_transformations=False,
    int_filtering=False):

    patch_utils.init_model_patch(model = model, mode = "test", seed_patch = patch)

    #create set of of spatial areas
    patch_areas = get_spatial_areas(
        patch_size = (patch.size(2), patch.size(3)),
        image_size = (1024, 2048),
        patch_pos = (None, None),
        num_areas = num_areas
    )
    
    patch_borders = get_borders(
        patch_size = (patch.size(2), patch.size(3)),
        image_size = (1024, 2048),
        patch_pos = (None, None),
        num_areas = num_areas
    )
    
    dict_spatial  = {}
    dict_spatial_patched = {}
    dict_spatial_no_patched = {}
    dict_mean = {}
    dict_std = {}
    dict_spatial_max = {}
    
    for layer in layers: 
        dict_spatial[layer]  = []
        dict_mean[layer] = []
        dict_std[layer] = []
        dict_spatial_max[layer] = []
        dict_spatial_patched[layer] = []
        dict_spatial_no_patched[layer] = []

    for idx, batch in enumerate(dataloader):
        images = batch[0].to('cuda')
        perturbed_image, _ = patch_utils.add_patch_to_batch(
            images = images.clone(), 
            patch = model.patch,
            patch_params = patch_params, 
            device = device, 
            int_filtering = int_filtering
        )

        activations_clear = get_activations(model, layers, images)
        activations_pert = get_activations(model, layers, perturbed_image)

        
        for layer in layers:
            check_list = isinstance(activations_clear['activations'][layer][0], list) or type(activations_clear['activations'][layer][0]) is tuple
            
            if check_list: 
                num_sub_layers = len(activations_clear['activations'][layer][0])   
                print(num_sub_layers)
            else:
                num_sub_layers = 1
                
            if len(dict_spatial[layer]) != num_sub_layers:
                for a in range(num_sub_layers):
                    dict_spatial[layer].append([])
                    dict_spatial_max[layer].append([])
                    dict_mean[layer].append([])
                    dict_std[layer].append([])
                    dict_spatial_patched[layer].append([])
                    dict_spatial_no_patched[layer].append([])
            
            # iterate for each sub_layers in layers
            for k in range(num_sub_layers):
                
                if num_sub_layers == 1:
                    act1 = activations_clear['activations'][layer][0]
                    act2 = activations_pert['activations'][layer][0]
                else:
                    act1 = activations_clear['activations'][layer][0][k]
                    act2 = activations_pert['activations'][layer][0][k]
                    
                            
                # iterate for each batch in sub_layers
                for im_idx in range(len(act1)):
                    heat1 = act1[im_idx]
                    mean, std = heat1.mean(axis=0), heat1.std(axis=0)
                    #area_ = cv2.resize(area,dsize=(heat1.shape[1], heat1.shape[0]))
                    heat1 = (heat1 - mean) / std
                    
                    heat2 = act2[im_idx]
                    heat2 = (heat2 - mean) / std
                    heat3 = heat1 - heat2
                    heat3 = np.abs(heat3)
                    

                    if(len(heat3.shape) > 2):
                        heat3 = np.mean(heat3, axis=0)
                        
                    if(len(heat1.shape) > 2):
                        #heat1 = heat1*std + mean
                        heat1 = np.abs(heat1)
                        heat1 = np.mean(heat1, axis=0)
                
                    if(len(heat2.shape) > 2):
                        #heat2 = heat2*std + mean
                        #heat2 = np.abs(heat2)
                        heat2 = np.abs(heat2)
                        heat2 = np.mean(heat2, axis=0)
                
                    attention_flag = False
                    if len(heat3.shape) == 1:
                        attention_flag = True
                        break
                        
                    spatial_values = []
                    mean_values = []
                    std_values = []
                    spatial_values_max = []
                    spatial_patched = []
                    spatial_no_patched = []
                    
                    if attention_flag is False:
                        for area in patch_areas:
                            area_ = cv2.resize(area, 
                                               dsize=(heat3.shape[1], heat3.shape[0]))
                            heat3_ = heat3*area_
                            spatial_values.append(heat3_.sum()/area_.sum())
                            spatial_values_max.append(heat3_.max())
                            
                            # NON è mean_values (CMABIARE!!)
                            mean_values.append(np.abs(heat1[area_==True].std()))
                            std_values.append(np.abs(heat2[area_==True].std()))
                            
                            heat2_ = heat2*area_
                            spatial_patched.append(heat2_.sum()/area_.sum())
                            
                            heat1_ = heat1*area_
                            spatial_no_patched.append(heat1_.sum()/area_.sum())
                        
                    spatial_values = np.array(spatial_values)    
                    dict_spatial[layer][k].append(spatial_values)
                    
                    spatial_patched = np.array(spatial_patched)
                    dict_spatial_patched[layer][k].append(spatial_patched)
                    
                    spatial_no_patched = np.array(spatial_no_patched)
                    dict_spatial_no_patched[layer][k].append(spatial_no_patched)
                    
                    mean_values = np.array(mean_values)   
                    dict_mean[layer][k].append(mean_values)
                    
                    std_values = np.array(std_values)   
                    dict_std[layer][k].append(std_values)
                    
                    spatial_values_max = np.array(spatial_values_max)
                    dict_spatial_max[layer][k].append(spatial_values_max)
                 
                if attention_flag is True:
                    break
            
                                                
    return dict_spatial, dict_spatial_max, dict_spatial_patched, dict_mean, dict_std, dict_spatial_no_patched, patch_areas, patch_borders


    


                    

                    





    
    

