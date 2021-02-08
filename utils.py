from metrics import metrics
import numpy as np

def metrics_multithreshold(Thresholds, predict_probs, y_true, i):
    
    y_pred = np.zeros_like(predict_probs[:,1])
    y_pred[predict_probs[:,1] >= Thresholds[i]] = 1
    y_pred[predict_probs[:,1] <  Thresholds[i]] = 0 

    accu, F1, Prec, R, Iou, Alarm_Area, tn, fp, fn, tp = metrics(y_true, y_pred)
    print(i, Thresholds[i])
    return accu, F1, Prec, R, Iou, Alarm_Area, tn, fp, fn, tp

def do(Thresholds, predict_probs, y_true):

    # Multiprocessing of thresholds.
    import multiprocessing
    from functools import partial
    n_cores = multiprocessing.cpu_count()
    p = multiprocessing.Pool(n_cores)
    func = partial(metrics_multithreshold, Thresholds, predict_probs, y_true )
    metrics_list = p.map(func, range(len(Thresholds)))
    p.close()
    p.join()

    return metrics_list
