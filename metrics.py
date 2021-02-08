#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:19:32 2020

@author: daliana
"""

from keras import backend as K
from sklearn.metrics import confusion_matrix
import numpy as np
from change_detection_with_DA import network, load_im, Normalization, create_mask, Using_buffer, split_tail
from skimage.morphology import area_opening
from reconstruction import Image_reconstruction, gray2rgb, Visualize_activations, Visualization_heatmaps, get_class_weights
import sys
import cv2
from sklearn.utils import shuffle
import math
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Model

# import matplotlib.image as mpimg
# import os
# import statistics

def metrics(y_true, y_pred):
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # print(tn, fp, fn, tp)
       
    intersection = np.sum(np.logical_and(y_pred, y_true)) # Logical AND  
    union = np.sum(np.logical_or(y_pred, y_true)) # Logical OR 
        
    accu = (tp + tn)/(tn + fp + fn + tp)
    Prec = tp/(tp + fp)
    R = tp/(tp + fn)
    F1 = 2 * Prec* R/(Prec + R)
    Iou = intersection/union
    Alarm_Area = (tp + fp)/(tn + fp + fn + tp)
    
    return accu, F1, Prec, R, Iou, Alarm_Area, tn, fp, fn, tp
    

def copy_txt(prefix, accu, F1, Prec, R, IoU, AA, tn, fp, fn, tp):  
               
    lt = 'a' 
    if not k:
        lt = 'w'
    if args.Arq == 1:
        file_metrics = open(prefix + "_metrics_Segnet.txt", lt)
    elif args.Arq == 2:
        file_metrics = open(prefix +"_metrics_Unet.txt", lt)
    elif args.Arq == 3:
        file_metrics = open(prefix +"_metrics_DeepLab.txt", lt)
    else:
        file_metrics = open(prefix +"_metrics_FCDenseNet.txt", lt)
    
    file_metrics.write('K-Fold:%d\n'%(k))
    file_metrics.write('Acc:%2f\n'%(100*accu))
    file_metrics.write('F1:%2f\n'%(100*F1))
    file_metrics.write('Recall:%2f\n'%(100*R))
    file_metrics.write('Precision:%2f\n'%(100*Prec))
    file_metrics.write('IoU:%2f\n'%(100*Iou))
    file_metrics.write('Alarm Area:%.2f\n\n'%(100*Alarm_Area))
    
    file_metrics.write('Confusion_matrix\n\n')
    file_metrics.write('TN:%2f\n\n'%(tn))
    file_metrics.write('FP:%2f\n\n'%(fp))
    file_metrics.write('FN:%2f\n\n'%(fn))
    file_metrics.write('TP:%2f\n\n'%(tp))      


if __name__=='__main__':

    from arguments import Arguments_metrics
    args = Arguments_metrics()

    if args.dataset == "Landsat8": _6ha_area = 69
    elif args.dataset == "Sentinel2": _6ha_area = 625

    overlap = round(args.patch_size * args.overlap_percent)
    overlap -= overlap % 2
    stride = args.patch_size - overlap

    if args.Mask_P_M:
        no_tiles_h, no_tiles_w = 10, 10
    else: 
        no_tiles_h, no_tiles_w = 5, 5

    see_intermediate_layers = 0

    x, files_name = load_im(args.dataset)

    for i in range(len(x)):
        if i <= 1:
            if args.dataset == "Landsat8":
                x[i] = x[i][:,1:2551,1:5121]
            elif args.dataset == "Sentinel2":
                x[i] = x[i][:,20:10930,:11000]
        else:
            if args.dataset == "Landsat8":
                x[i]  = x[i][1:2551,1:5121]
            elif args.dataset == "Sentinel2":         
                x[i]  = x[i][20:10930,:11000]

    I = np.concatenate((x[0], x[1]), axis = 0) 
    I = I.transpose((1, 2, 0))
    nChannels = I.shape[-1]
    
    rows, cols, c = I.shape
    Norm_image = Normalization(I, args.Arq)

    past_reference = x[3]
    act_reference = x[2] 

    del I, x  # saving memory

    reference = act_reference.copy()

    class_deforestation = 0
    class_background = 0

    _, counts = np.unique(act_reference, return_counts=True)
    print(counts)
    class_deforestation += counts[1]
    class_background += counts[0]
    print(' Class Deforestation_No.pixeles:%2f' %(class_deforestation))
    print(' Class Background_No.pixeles:%2f' %(class_background))
    print(' Percent Class Deforestation:%2f' %(class_deforestation * 100/(class_background + class_deforestation))) 
    print(' Percent Class Background: %2f' %(class_background * 100/(class_background + class_deforestation)))  
    print(' Proporcion:%2f' %(class_deforestation / class_background))  

    # Cancel buffer
    if args.cancel_buffer:
        reference = Using_buffer(reference)
        if len(args.weights) == 2:
            args.weights.append(0)
        print('Cancel Buffer')
    
    # Cancel past reference
    reference[past_reference == 1] = 2
    if len(args.weights) == 2:
        args.weights.append(0)
    print('Cancel Past Reference')
    
    # Split Image in Tails
    tiles_Image = split_tail(rows, cols, no_tiles_h, no_tiles_w)
    if args. Mask_P_M: 
        Train_tiles = np.array([2, 6, 13, 24, 28, 35, 37, 46, 47, 53, 58, 60, 64, 71, 75, 82, 86, 88, 93])
        Valid_tiles = np.array([8, 11, 26, 49, 78])
        
        Test_tiles = np.arange(100)  
        Test_tiles = list(set(Test_tiles) - set(Train_tiles))
        Test_tiles = list(set(Test_tiles) - set(Valid_tiles))
        
        train_set =[]
        val_set = []
        test_set = []
        
        for i in Train_tiles:
            train_set.append(tiles_Image[i-1])
        
        for i in Valid_tiles:
            val_set.append(tiles_Image[i-1])
            
        for i in Test_tiles:
            test_set.append(tiles_Image[i-1])

    accuracy, f1_score, precision, recall, IoU, AA = (np.zeros((args.Npoints)) for i in range(6))
    _accu, _F1 , _Prec, _R, _Iou, _tn, _fp, _fn, _tp = ([] for i in range(9))

    for k in range(args.N_run):
        
        # if k_fold:
        #     print("No-k fold %d" %(k))
        #     tiles_Image = shuffle(tiles_Image, random_state = k)
            
        #     test_set = tiles_Image[size * k: size * (k+1)]
        #     train_set = tiles_Image[: size * k] + tiles_Image[size * (k + 1) :]            
        #     val_set = train_set[: math.ceil(0.1 * len(train_set))]
            
        # if cross_validation:
        #     print("Run number %d" %(k))
        #     tiles_Image = shuffle(tiles_Image, random_state = k)
            
        #     train_set = tiles_Image[: math.ceil(0.25 * len(tiles_Image))]
        #     val_set = train_set[: math.ceil(0.05 * len(train_set))]
        #     train_set = train_set[math.ceil(0.05 * len(train_set)):]
        #     test_set = tiles_Image[math.ceil(0.25 * len(tiles_Image)) :]
                

        mask = create_mask(rows, cols, val_set, test_set, no_tiles_h, no_tiles_w )
                
        net = network(args.Arq, reference, args.weights, args.patch_size, nChannels, args) 
        
        if args.Arq == 1:
            net.load_weights('best_model_Segnet_%d.h5'%(k))
        elif args.Arq == 2:
            net.load_weights('best_model_Unet_%d.h5'%(k))
        elif args.Arq == 3:
            net.load_weights('best_model_Deep_%d.h5'%(k))
        elif args.Arq == 4:
            net.load_weights('best_model_Dense_%d.h5'%(k))
        else:
            print("specify architecture")
            import sys; sys.exit()
        
        net.summary()               
        obj = Image_reconstruction(net, 3, patch_size = args.patch_size, overlap_percent = args.overlap_percent)
            
        # Prediction stage
        print("Run number %d" %(k))
        predict_probs = obj.Inference(Norm_image)

        try:
            mean_probs += predict_probs / args.N_run
        except NameError:
            mean_probs = predict_probs / args.N_run
        
        predict_labels = predict_probs.argmax(axis=-1)
        predict_labels[predict_labels == 2] = 0

        # Test mask
        test_mask = np.zeros_like(mask)
        test_mask[mask == 0] = 1    # This is to chose the test region of the mask
        ######## Not taking into account the past desforestation
        test_mask[reference == 2] = 0
                        
        less_6ha_predict = predict_labels - area_opening(predict_labels.astype('int'),
                                            area_threshold = _6ha_area, connectivity=1)
        test_mask[less_6ha_predict == 1] = 0
    
        ################ If I want to remove the area of 69 pixeles in the reference ####################################           
        if args.remove_69_ref:      
            less_6ha_ref = act_reference - area_opening(act_reference.astype('int'),
                                                area_threshold = _6ha_area, connectivity=1)
            test_mask[less_6ha_ref == 1] = 0
            print(np.unique(less_6ha_ref))

        y_true = act_reference[test_mask == 1]   
        y_pred = predict_labels[test_mask == 1]
        accu, F1, Prec, R, Iou, Alarm_Area, tn, fp, fn, tp = metrics(y_true, y_pred)
        copy_txt("Models",accu, F1, Prec, R, IoU, AA, tn, fp, fn, tp)
        _accu.append(accu)
        _F1.append(F1)
        _Prec.append(Prec)
        _R.append(R)
        _Iou.append(Iou)
        _tn.append(tn)
        _fp.append(fp)
        _fn.append(fn)
        _tp.append(tp)

        # Multiprocessing thresholds.
        Pmax = np.max(predict_probs[:,:,1][act_reference * test_mask == 1])
        Thresholds = np.linspace(Pmax, 0, args.Npoints)
        
        predict_probs = predict_probs[test_mask == 1]
        del less_6ha_predict, net, obj, y_pred  # saving memory

        from utils import do
        metrics_list = do(Thresholds, predict_probs, y_true)

        metrics_list = np.asarray(metrics_list).transpose()
        accuracy  += metrics_list[0, :] / args.N_run
        f1_score  += metrics_list[1, :] / args.N_run
        precision += metrics_list[2, :] / args.N_run
        recall    += metrics_list[3, :] / args.N_run
        IoU       += metrics_list[4, :] / args.N_run
        AA        += metrics_list[5, :] / args.N_run
            
    ################## MEAN +- STD ######################
    _accu = np.asarray(_accu)
    _F1 = np.asarray(_F1)
    _Prec = np.asarray(_Prec)
    _R = np.asarray(_R)
    _Iou = np.asarray(_Iou)
    _tn = np.asarray(tn)
    _fp = np.asarray(fp)
    _fn = np.asarray(fn)
    _tp = np.asarray(tp)

    lt = 'a' 
    if args.Arq == 1:
        file_metrics = open("Models" + "_metrics_Segnet.txt", lt)
    elif args.Arq == 2:
        file_metrics = open("Models" +"_metrics_Unet.txt", lt)
    elif args.Arq == 3:
        file_metrics = open("Models" +"_metrics_DeepLab.txt", lt)
    else:
        file_metrics = open("Models" +"_metrics_FCDenseNet.txt", lt)
    
    file_metrics.write('----------- MEAN +- STD ---------\n')
    file_metrics.write('Acc: %2f +- %2f\n'%(100*_accu.mean(), 100*_accu.std()))
    file_metrics.write('F1: %2f +- %2f\n'%(100*_F1.mean(), 100*_F1.std()))
    file_metrics.write('Prec: %2f +- %2f\n'%(100*_Prec.mean(), 100*_Prec.std()))
    file_metrics.write('R: %2f +- %2f\n'%(100*_R.mean(), 100*_R.std()))
    file_metrics.write('Iou: %2f +- %2f\n'%(100*_Iou.mean(), 100*_Iou.std()))
    file_metrics.write('tn: %d +- %d\n'%(_tn.mean(), _tn.std()))
    file_metrics.write('fp: %d +- %d\n'%(_fp.mean(), _fp.std()))
    file_metrics.write('fn: %d +- %d\n'%(_fn.mean(), _fn.std()))
    file_metrics.write('tp: %d +- %d\n'%(_tp.mean(), _tp.std()))

    np.save("Threshold_mean_curves_%d_%d"%(args.Arq, k),Thresholds)
    np.save("Recall_mean_curves_%d_%d"%(args.Arq, k),recall)
    np.save("Precision_mean_curves_%d_%d"%(args.Arq, k),precision)
    np.save("Alarm Area_mean_curves_%d_%d"%(args.Arq, k), AA)       

    ##### Only to see the image the test part with the actual deforestation in the prediction and in the reference
    predict_total = predict_labels * test_mask
    test_reference = act_reference * test_mask
    predict_see = gray2rgb(np.uint(predict_total))
    reference_see = gray2rgb(np.uint(test_reference))
    cv2.imwrite('./predict_total_%d.tiff'%(args.Arq), predict_see)
    cv2.imwrite('./test_reference.tiff', reference_see)
    test_mask_see = gray2rgb(np.uint(test_mask))
    cv2.imwrite('./test_mask.tiff', test_mask_see)


    # ############## Mean Prediction metrics ###################
    # predict_labels = mean_probs.argmax(axis=-1)
    # predict_labels[predict_labels == 2] = 0
            
    # # Test mask
    # test_mask = np.zeros_like(mask)
    # test_mask[mask == 0] = 1    # This is to choose the test region of the mask
    # ######## Not taking into account the past desforestation
    # test_mask[reference == 2] = 0
                    
    # less_6ha_predict = predict_labels - area_opening(predict_labels.astype('int'),
    #                                     area_threshold = _6ha_area, connectivity=1)
    # test_mask[less_6ha_predict == 1] = 0

    # ################ If I want to remove the area of 69 pixeles in the reference ####################################           
    # if args.remove_69_ref:      
    #     less_6ha_ref = act_reference - area_opening(act_reference.astype('int'),
    #                                         area_threshold = _6ha_area, connectivity=1)
    #     test_mask[less_6ha_ref == 1] = 0
    #     print(np.unique(less_6ha_ref))
    
    # y_true = act_reference[test_mask == 1]   
    # y_pred = predict_labels[test_mask == 1]
        
    # accu, F1, Prec, R, Iou, Alarm_Area, tn, fp, fn, tp = metrics(y_true, y_pred)
    # copy_txt("Mean_map",accu, F1, Prec, R, IoU, AA, tn, fp, fn, tp)
        
    # # Multiprocessing thresholds.
    # Pmax = np.max(mean_probs[:,:,1][act_reference * test_mask == 1])
    # Thresholds = np.linspace(Pmax, 0, args.Npoints)
    
    # mean_probs = mean_probs[test_mask == 1]
    # del less_6ha_predict, Norm_image, mask # saving memory

    # from utils import do
    # metrics_list = do(Thresholds, mean_probs, y_true)

    # metrics_list = np.asarray(metrics_list).transpose()
    # accuracy  = metrics_list[0, :]
    # f1_score  = metrics_list[1, :]
    # precision = metrics_list[2, :]
    # recall    = metrics_list[3, :]
    # IoU       = metrics_list[4, :]
    # AA        = metrics_list[5, :]

    # np.save("Threshold_mean_map_%d_%d"%(Arq, k),Thresholds)
    # np.save("Recall_mean_map_%d_%d"%(Arq, k),recall)
    # np.save("Precision_mean_map_%d_%d"%(Arq, k),precision)
    # np.save("Alarm Area_mean_map_%d_%d"%(Arq, k), AA)       
    
    
    ###################################################
        #plt.plot(np.array(recall)*100, np.array(AA)*100)
        ## naming the x axis 
        #plt.xlabel('Recall (%)') 
        ## naming the y axis 
        #plt.ylabel('Alarm Area (%)') 
        #plt.xticks([80, 85, 90, 95, 100])
        #plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
        #plt.grid(True)
        ## giving a title to my graph 
        #plt.title('Alarm Area vs Recall')   
        #plt.show()
        

        # =============================================================================            
        ######################## See the activation of intermediate layers #####################           
        #        if see_intermediate_layers:
        #            test_patches = []
        #            patch_im_test = []
        #            patch_gdt_test = []
        #
        #            step_row = (stride - rows % stride) % stride
        #            step_col = (stride - cols % stride) % stride
        #            k1, k2 = (rows + step_row)//stride, (cols + step_col)//stride
        #            
        #            for i in range(k1):
        #                for j in range(k2):
        #                    # Test
        #                    if test_mask[i*stride:i*stride + args.patch_size, j*stride:j*stride + args.patch_size].all():
        #                        test_patches.append((i*stride, j*stride)) 
        #            
        #            pad_tuple_img = ( (overlap//2, overlap//2 + step_row), (overlap//2, overlap//2 + step_col), (0, 0) )
        #            pad_tuple_msk = ( (overlap//2, overlap//2 + step_row), (overlap//2, overlap//2 + step_col) )
        #            img_pad = np.pad(Norm_image, pad_tuple_img, mode = 'symmetric')
        #            gdt_pad = np.pad(reference, pad_tuple_msk, mode = 'symmetric')
        #            
        ##            Patch of Images
        #            I_patch = img_pad[test_patches[48][0]: test_patches[48][0] + args.patch_size, test_patches[48][1]: test_patches[48][1] + patch_size,:]
        #            patch_im_test.append(I_patch)  
        #            
        ##           Patch of References                    
        #            gdt_patch = gdt_pad[test_patches[48][0]: test_patches[48][0] + args.patch_size, test_patches[48][1]: test_patches[48][1] + patch_size]
        #            patch_gdt_test.append(gdt_patch) 
        #            
        #            patch_im_test = np.array(patch_im_test)
        #            patch_gdt_test = np.array(patch_gdt_test)
        ##            print(np.array(patch_im_test).shape)
        #           
        ##            plt.figure()
        ##            imgplot = plt.imshow(patch_im_test[:,:,:,[3,2,1]][0])
        ##            plt.show()   
        ##
        ##            plt.figure()
        ##            imgplot = plt.imshow(patch_im_test[:,:,:,[3+8,2+8,1+8]][0])
        ##            plt.show()   
        ##            
        ##            plt.figure()
        ##            imgplot = plt.imshow(patch_gdt_test[0])
        ##            plt.show()   
        #
        #            layer_name =  'conv_14' 
        #            # intermediate layer, get output features
        #            layer = Model(inputs=net.input, outputs=net.get_layer(layer_name).output)
        #            
        ##            This is the activation for thw fourth convolution layer for a patch of the deforestation image.           
        #            layer_activations = layer.predict(patch_im_test)
        #            layer_activations = layer_activations[0]           
        ##            print(layer_activations.shape)
        ##            plt.matshow(layer_activations[0,:,:,10], cmap ='viridis')
        #            
        ##            Visualizing every channel of a layer:
        #            Visualize_activations(layer_name, layer_activations)
        ## =============================================================================            
        #            # Visualize weights for layer 
        #            W = net.get_layer(name=layer_name).get_weights()[0]
        #            W = np.squeeze(W)
        #            W = W[:,:,0,:]
        #
        #            npad = ((1,1), (1,1), (0,0))
        #            W = np.pad(W, pad_width = npad, mode='constant', constant_values=0)
        #            
        #            Visualize_activations(layer_name, W)
        #          
        ## =============================================================================          
        ##            Using grad-cam technique:
        #            heatmap = Visualization_heatmaps(layer_name, layer_activations, net, patch_im_test)
        ## =============================================================================
        #            #Resizes the heatmap to be the same size as the original image
        #            up_heatmap = cv2.resize(heatmap,(patch_im_test.shape[1], patch_im_test.shape[2]))
        #            
        #            ##Convert it to RGB
        #            up_heatmap =np.uint8(255 * up_heatmap)
        #            
        #            plt.imshow(patch_im_test)
        #            plt.imshow(up_heatmap, alpha = 0.5)
        #            plt.show()   
        
        # ============================================================================= 
        # =============================================================================
        
        
        # =============================================================================

                        
    #            print('Test accuracy:%.2f' %(100*accu))
    #            print('Test f1score:%.2f' %(100*F1))
    #            print('Test prescision:%.2f' %(100*Prec))
    #            print('Test recall:%.2f' %(100*R))
    #            print('Intersection over Union:%.2f' %(100*Iou))
    #            print('Alarm Area:%.2f'%(100*Alarm_Area))
