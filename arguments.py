import argparse

def Arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--Arq', dest='Arq', type=int, default=2, help='Architecture: 1 - Segnet, 2 - Unet, 3 - DeepLab, 4 - DenseNet')
    parser.add_argument('--dataset', type=str, choices=['Landsat8', 'Sentinel2'], default='Landsat8', help='dataset name')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=128, help= 'dimension of the extracted patches')
    parser.add_argument('--overlap_percent', dest='overlap_percent', type=float, default=0.8, help= 'overlap between classes')

    parser.add_argument('--cancel_buffer', dest='cancel_buffer', type=eval, choices=[True, False], default=True, help='Decide wether a buffer around deforestated regions will be performed')
    
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for the source classifier')
    parser.add_argument('--N_run', dest='N_run', type=int, default=1, help='number of executions of the algorithm')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--patience', dest='patience', type=int, default=10, help='number of epochs without improvement to apply early stop')
    parser.add_argument('--Mask_P_M', dest='Mask_P_M', type=eval, choices=[True, False], default=True, help='mask')
    parser.add_argument('--weights', dest='weights', type=list, default=[0.4, 2], help='weights for weighted cross entropy loss classes: [0, 1]')

    args = parser.parse_args()

    return args

def Arguments_metrics():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--Arq', dest='Arq', type=int, default=2, help='Architecture: 1 - Segnet, 2 - Unet, 3 - DeepLab, 4 - DenseNet')
    parser.add_argument('--Npoints', dest='Npoints', type=int, default=200, help='number of points in the curves pr vs rec')
    parser.add_argument('--dataset', type=str, choices=['Landsat8', 'Sentinel2'], default='Landsat8', help='dataset name')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=128, help= 'dimension of the extracted patches')
    parser.add_argument('--overlap_percent', dest='overlap_percent', type=float, default=0.8, help= 'overlap between classes')

    parser.add_argument('--cancel_buffer', dest='cancel_buffer', type=eval, choices=[True, False], default=True, help='Decide wether a buffer around deforestated regions will be performed')
    parser.add_argument('--remove_69_ref', dest='remove_69_ref', type=eval, choices=[True, False], default=False, help='remove 6ha area from the reference')
    
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for the source classifier')
    parser.add_argument('--N_run', dest='N_run', type=int, default=1, help='number of executions of the algorithm')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--Mask_P_M', dest='Mask_P_M', type=eval, choices=[True, False], default=True, help='mask')
    parser.add_argument('--weights', dest='weights', type=list, default=[0.4, 2], help='weights for weighted cross entropy loss classes: [0, 1]')

    args = parser.parse_args()

    return args
