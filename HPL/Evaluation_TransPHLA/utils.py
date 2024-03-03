from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)

# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()


def performances_to_pd(performances_list):
    metrics_name = ['roc_auc', 'accuracy', 'mcc', 'f1',
                    'sensitivity', 'specificity', 'precision', 'recall', 'aupr']

    performances_pd = pd.DataFrame(performances_list, columns=metrics_name)
    performances_pd.loc['mean'] = performances_pd.mean(axis=0)
    performances_pd.loc['std'] = performances_pd.std(axis=0)

    return performances_pd


def f_mean(l): return sum(l)/len(l)


def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])


def performances(y_true, y_pred, y_prob, print_=True):

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]).ravel().tolist()
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    try:
        mcc = ((tp*tn) - (fn*fp)) / \
            np.sqrt(np.float((tp+fn)*(tn+fp)*(tp+fp)*(tn+fn)))
    except:
        print('MCC Error: ', (tp+fn)*(tn+fp)*(tp+fp)*(tn+fn))
        mcc = np.nan
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    try:
        recall = tp / (tp+fn)
    except:
        recall = np.nan

    try:
        precision = tp / (tp+fp)
    except:
        precision = np.nan

    try:
        f1 = 2*precision*recall / (precision+recall)
    except:
        f1 = np.nan

    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    if print_:
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: 0 = {} | 1 = {}'.format(
            Counter(y_pred)[0], Counter(y_pred)[1]))
        print('y_true: 0 = {} | 1 = {}'.format(
            Counter(y_true)[0], Counter(y_true)[1]))
        print('auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}'.format(
            roc_auc, sensitivity, specificity, accuracy, mcc))
        print('precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}'.format(
            precision, recall, f1, aupr))
        # d = pd.DataFrame([[roc_auc, accuracy, mcc, f1]])
        # d.to_csv('result.csv', mode='a', header=False, index=None)
    return (roc_auc, accuracy, mcc, f1, sensitivity, specificity, precision, recall, aupr)


def generate_hla_rank(train_data):
    train_set_stat = train_data.groupby(by='HLA').count()
    train_set_stat = train_set_stat.sort_values(by='peptide', ascending=False)
    return list(train_set_stat.index)


# def save_to_tensorboard(metrics_train, loss_train_list, metrics_val, loss_val_list, epoch):
#     writer.add_scalar('metric/train/auc', metrics_train[0], epoch)
#     writer.add_scalar('metric/train/acc', metrics_train[1], epoch)
#     writer.add_scalar('metric/train/mcc', metrics_train[2], epoch)
#     writer.add_scalar('metric/train/f1', metrics_train[3], epoch)
#     writer.add_scalar('metric/train/avg', sum(metrics_train[:4])/4, epoch)
#     writer.add_scalar('metric/train/loss', f_mean(loss_train_list), epoch)

#     writer.add_scalar('metric/val/auc', metrics_val[0], epoch)
#     writer.add_scalar('metric/val/acc', metrics_val[1], epoch)
#     writer.add_scalar('metric/val/mcc', metrics_val[2], epoch)
#     writer.add_scalar('metric/val/f1', metrics_val[3], epoch)
#     writer.add_scalar('metric/val/avg', sum(metrics_val[:4])/4, epoch)
#     writer.add_scalar('metric/val/loss', f_mean(loss_val_list), epoch)


# def save_test_to_tensorboard(metrics_test, loss_test_list, epoch):
#     writer.add_scalar('metric/test/auc', metrics_test[0], epoch)
#     writer.add_scalar('metric/test/acc', metrics_test[1], epoch)
#     writer.add_scalar('metric/test/mcc', metrics_test[2], epoch)
#     writer.add_scalar('metric/test/f1', metrics_test[3], epoch)
#     writer.add_scalar('metric/test/avg', sum(metrics_test[:4])/4, epoch)
#     writer.add_scalar('metric/test/loss', f_mean(loss_test_list), epoch)


