# -*- coding: utf-8 -*-
import os
import time


import matplotlib.pyplot as plt


def a_l_plot(acc, val_acc, loss, val_loss, modelType, save_path, height, *args):
    ''':argument'''

    epochs = range(0, len(acc))

    plt.figure(num=0, figsize=(15, 10))
    plt.title('Training and Validation Accuracy', fontsize=20)
    plt.xlabel(u'Iteration', fontsize=14)
    plt.ylabel(u'Accuracy', fontsize=14)
    plt.grid(True)
    plt.ylim(0.75, 1.00)
    plt.xlim(0, height)
    plt.plot(epochs, acc, args[0][0], label=args[0][1])
    plt.plot(epochs, val_acc, args[1][0], label=args[1][1])
    plt.legend(loc=4)
    now_time = time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='y', m='m', d='d', h='h', M='Min', s='s')
    img_path = os.path.join(save_path, 'acc_' + modelType + now_time)
    img_path = img_path + '.tif'
    plt.savefig(img_path)

    plt.figure(num=1, figsize=(15, 10))
    plt.title('Training and Validation Loss', fontsize=20)
    plt.ylabel(u'Loss', fontsize=14)
    plt.xlabel(u'Iteration', fontsize=14)
    plt.grid(True)
    plt.ylim(0.00, 1.00)
    plt.xlim(0, height)
    plt.plot(epochs, loss, args[2][0], label=args[2][1])
    plt.plot(epochs, val_loss, args[3][0], label=args[3][1])
    plt.legend(loc=1)
    now_time = time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='y', m='m', d='d', h='h', M='Min', s='s')
    img_path = os.path.join(save_path, 'loss_' + modelType + now_time)
    img_path = img_path + '.tif'
    plt.savefig(img_path)
