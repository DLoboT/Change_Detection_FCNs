import os

exc = []

############## LANDSAT-8 ###############
# Segnet
exc.append("python change_detection_with_DA.py  --dataset Landsat8 --Arq 1 --N_run 10 --overlap_percent 0.8 --patch_size 128")
exc.append("python metrics.py                   --dataset Landsat8 --Arq 1 --N_run 10 --overlap_percent 0.5 --patch_size 128")
# Unet
exc.append("python change_detection_with_DA.py  --dataset Landsat8 --Arq 2 --N_run 10 --overlap_percent 0.8 --patch_size 128 ")
exc.append("python metrics.py                   --dataset Landsat8 --Arq 2 --N_run 10 --overlap_percent 0.5 --patch_size 128")
# DeepLab
exc.append("python change_detection_with_DA.py  --dataset Landsat8 --Arq 3 --N_run 10 --overlap_percent 0.8 --patch_size 128")
exc.append("python metrics.py                   --dataset Landsat8 --Arq 3 --N_run 10 --overlap_percent 0.5 --patch_size 128")
# DenseNet
exc.append("python change_detection_with_DA.py  --dataset Landsat8 --Arq 4 --N_run 10 --overlap_percent 0.8 --patch_size 128")
exc.append("python metrics.py                   --dataset Landsat8 --Arq 4 --N_run 10 --overlap_percent 0.5 --patch_size 128")

############## SENTINEL-2 ###############
# Segnet
exc.append("python change_detection_with_DA.py  --dataset Sentinel2 --Arq 1 --N_run 10 --overlap_percent 0.8 --patch_size 128")
exc.append("python metrics.py                   --dataset Sentinel2 --Arq 1 --N_run 10 --overlap_percent 0.5 --patch_size 128")
# Unet
exc.append("python change_detection_with_DA.py  --dataset Sentinel2 --Arq 2 --N_run 10 --overlap_percent 0.8 --patch_size 128")
exc.append("python metrics.py                   --dataset Sentinel2 --Arq 2 --N_run 10 --overlap_percent 0.5 --patch_size 128")
# DeepLab
exc.append("python change_detection_with_DA.py  --dataset Sentinel2 --Arq 3 --N_run 10 --overlap_percent 0.8 --patch_size 128")
exc.append("python metrics.py                   --dataset Sentinel2 --Arq 3 --N_run 10 --overlap_percent 0.5 --patch_size 128")
# DenseNet
exc.append("python change_detection_with_DA.py  --dataset Sentinel2 --Arq 4 --N_run 10 --overlap_percent 0.8 --patch_size 128")
exc.append("python metrics.py                   --dataset Sentinel2 --Arq 4 --N_run 10 --overlap_percent 0.5 --patch_size 128")

if __name__=='__main__':
    for i in exc:
        os.system(i)
