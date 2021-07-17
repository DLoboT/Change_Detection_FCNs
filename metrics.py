#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:19:32 2020

@author: daliana
"""

from keras import backend as K
from sklearn.metrics import confusion_matrix
import numpy as np
from change_detection_with_DA import network, load_im, Normalization, create_mask, Using_buffer, split_tail, Hot_encoding
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

def loss(y_true, y_pred, weights):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= np.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = np.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * np.log(y_pred) * weights
    loss = -np.sum(loss, -1)
    return loss

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
        file_metrics = open(prefix +"_metrics_DeepLab_xception.txt", lt)
    elif args.Arq == 4:
        file_metrics = open(prefix +"_metrics_DeepLab_mobile.txt", lt)
    elif args.Arq == 5:
        file_metrics = open(prefix +"_metrics_FCDenseNet.txt", lt)
    else:
        file_metrics = open(prefix +"_metrics_ResUNet.txt", lt)

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
        no_tiles_h, no_tiles_w = 3, 5
    else: 
        no_tiles_h, no_tiles_w = 5, 5

    see_intermediate_layers = 0

    x, files_name = load_im(args.dataset)
    
    for i in range(len(x)):
        if i <= 1:

            if args.dataset == "Landsat8":
                x[i] = x[i][:,1510:,:3670]
            elif args.dataset == "Sentinel2":
               x[i] = x[i][:,3*1510:,:3*3670]
        else:
            if args.dataset == "Landsat8":
                 x[i]  = x[i][1510:,:3670]
            elif args.dataset == "Sentinel2":         
                x[i]  = x[i][3*1510:,:3*3670]

    I = np.concatenate((x[0], x[1]), axis = 0) 
    I = I.transpose((1, 2, 0))
    nChannels = I.shape[-1]
    
    rows, cols, c = I.shape
    Norm_image = Normalization(I)

    past_reference = x[3]
    act_reference = x[2] 
    print(act_reference.shape)
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
        reference = Using_buffer(reference, args)
        args.weights.append(0)
        print('Cancel Buffer')
    
    # Cancel past reference
    reference[past_reference == 1] = 2
    if len(args.weights) == 0:
        args.weights.append(0)
    print('Cancel Past Reference')
    
    # Split Image in Tails
    tiles_Image = split_tail(rows, cols, no_tiles_h, no_tiles_w)
    if args. Mask_P_M: 
        Train_tiles = np.array([13, 14, 15]) 
        Valid_tiles = np.array([9])
        Test_tiles = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12])
        
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

    mask = create_mask(rows, cols, train_set, val_set, test_set, no_tiles_h, no_tiles_w )
    # plt.figure()
    # imgplot = plt.imshow(mask)
    # plt.savefig('mask.png')

    # save_image = Image.fromarray(np.uint8(mask*255))    
    # save_image.save('mask_{}_{}.tiff'.format(args.Arq,args.dataset))
    # sys.exit()

    # Weights by (Patel,2020)
    # train_tile = act_reference[mask==1]
    # class_k, count = np.unique(train_tile, return_counts=True)
    # args.weights = [1] + args.weights
    # args.weights = [(np.log(sum(counts[:2])) - np.log(counts[0]))/ (np.log(sum(counts[:2])) - np.log(counts[1]))] + args.weights
    # args.weights = [element * 5 for element in args.weights]
    args.weights = [0.4, 2, 0]
    print(args.weights)

    for k in range(args.N_run):                 
               
        net = network(args.Arq, reference, args.weights, args.patch_size, nChannels, args) 
        
        if args.Arq == 1:
            net.load_weights('best_Model_Segnet_%d.h5'%(k))
        elif args.Arq == 2:
            net.load_weights('best_Model_Unet_%d.h5'%(k))
        elif args.Arq == 3:
            net.load_weights('best_Model_DeepLab_xception_%d.h5'%(k))
        elif args.Arq == 4:
            net.load_weights('best_Model_DeepLab_mobilenetv2_%d.h5'%(k))
        elif args.Arq == 5:
            net.load_weights('best_Model_FCDenseNet_%d.h5'%(k))
        elif args.Arq == 6:
            net.load_weights('best_Model_Resunet_%d.h5'%(k))
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
        # test_mask[mask == .5] = 1    # This is to chose the val region of the mask
        # test_mask[mask == 1] = 1    # This is to chose the train region of the mask
        
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
        Pmax = np.max(predict_probs[:,:,1][test_mask == 1])
        th1, th2 = 0.02, 0.0
        u1, u2 = 200, 1800
        Thresholds = np.concatenate((np.linspace(Pmax, th1, u1),
                                     np.linspace(th1 - 1e-6, th2, u2)))
        
        # Thresholds = np.linspace(Pmax, 0, args.Npoints)

        plt.figure(1)
        predict_probs_img = predict_probs[:,:,1] * test_mask  
        save_image = Image.fromarray(np.uint8(predict_probs_img*255))    
        save_image.save('map_probs_{}_{}.tiff'.format(args.Arq,args.dataset))
        # plt.imshow(np.abs(predict_probs_img),cmap='jet')
        # plt.colorbar()
        # plt.savefig('predict_probs_{}.tiff'.format(args.Arq), dpi=300, format='png', bbox_inches = 'tight')
        # plt.close()

        predict_probs = predict_probs[test_mask == 1]
        np.save("predict_probs_%d_%d"%(args.Arq, k),predict_probs)

        del less_6ha_predict, net, obj, y_pred  # saving memory
        # del net, obj, y_pred  # saving memory

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
    _tn = np.asarray(_tn)
    _fp = np.asarray(_fp)
    _fn = np.asarray(_fn)
    _tp = np.asarray(_tp)

    lt = 'a' 
    if args.Arq == 1:
        file_metrics = open("Models" + "_metrics_Segnet.txt", lt)
    elif args.Arq == 2:
        file_metrics = open("Models" +"_metrics_Unet.txt", lt)
    elif args.Arq == 3:
        file_metrics = open("Models" +"_metrics_DeepLab_xception.txt", lt)
    elif args.Arq == 4:
        file_metrics = open("Models" +"_metrics_DeepLab_mobile.txt", lt)
    elif args.Arq == 5:
        file_metrics = open("Models" +"_metrics_FCDenseNet.txt", lt)
    else:       
        file_metrics = open("Models" +"_metrics_ResUNet.txt", lt)
    
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
    np.save("Alarm_Area_mean_curves_%d_%d"%(args.Arq, k), AA)       

    ##### Only to see the image the test part with the actual deforestation in the prediction and in the reference
    predict_total = predict_labels * test_mask
    test_reference = act_reference * test_mask
    predict_see = gray2rgb(np.uint(predict_total))
    reference_see = gray2rgb(np.uint(test_reference))
    cv2.imwrite('./predict_total_%d.tiff'%(args.Arq), predict_see)
    cv2.imwrite('./test_reference.tiff', reference_see)
    test_mask_see = gray2rgb(np.uint(test_mask))
    cv2.imwrite('./test_mask.tiff', test_mask_see)

   
