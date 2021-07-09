# -*- coding: utf-8 -*-

from Module.CACP import CACP
from tensorflow.python.keras.backend import *
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.python.keras.models import *
from tensorflow.python.keras.optimizers import *


def BMFRNet(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.2)(conv4)
    conv5 = CACP(drop4, 1024)
    drop5 = Dropout(0.2)(conv5)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop5)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    pred1 = Conv2D(1, 1, activation='sigmoid', name="pred1")(conv6)
    lossLay1 = UpSampling2D(size=(8, 8), name="lossLay1", interpolation='bilinear')(pred1)
    pred1 = UpSampling2D(interpolation='bilinear')(pred1)

    up7 = Conv2D(256, 1, activation='relu', strides=1, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6))
    merge7 = concatenate([conv3, up7, pred1])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    pred2 = Conv2D(1, 1, activation='sigmoid', name="pred2")(conv7)
    lossLay2 = UpSampling2D(size=(4, 4), name="lossLay2", interpolation='bilinear')(pred2)
    pred2 = UpSampling2D(interpolation='bilinear')(pred2)

    up8 = Conv2D(128, 1, activation='relu', strides=1, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), interpolation='bilinear')(conv7))
    merge8 = concatenate([conv2, up8, pred2])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    pred3 = Conv2D(1, 1, activation='sigmoid', name="pred3")(conv8)
    lossLay3 = UpSampling2D(size=(2, 2), name="lossLay3", interpolation='bilinear')(pred3)
    pred3 = UpSampling2D(interpolation='bilinear')(pred3)

    up9 = Conv2D(64, 1, activation='relu', strides=1, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), interpolation='bilinear')(conv8))
    merge9 = concatenate([conv1, up9, pred3])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    result = Conv2D(1, 1, activation='sigmoid', name="result")(conv9)  # 预测结果4
    layers = [lossLay1, lossLay2, lossLay3]
    model = Model(inputs=inputs, outputs=result)
    myloss = Weighted_Diceloss(layers)
    model.compile(optimizer=Adam(lr=1e-4), loss=myloss, metrics=['accuracy'])
    model.summary()
    return model


def dice_loss(y_true_dice, y_pre_dice):
    smooth = 1.  # 1e-5
    y_true_f = K.flatten(y_true_dice)
    y_pred_f = K.flatten(y_pre_dice)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
    loss = 1 - score
    return loss


def Multi_level_Diceloss (y_true, y_pred, layers):
    lossPred0 = 0.1 * dice_loss(y_true_dice=y_true, y_pre_dice=layers[0])
    lossPred1 = 0.2 * dice_loss(y_true_dice=y_true, y_pre_dice=layers[1])
    lossPred2 = 0.3 * dice_loss(y_true_dice=y_true, y_pre_dice=layers[2])
    lossPredResult = 0.4 * dice_loss(y_true_dice=y_true, y_pre_dice=y_pred)
    sumLoss = lossPred0 + lossPred1 + lossPred2 + lossPredResult
    return sumLoss


def Weighted_Diceloss(layers):
    def dice(y_true, y_pred):
        return Multi_level_Diceloss(y_true, y_pred, layers)
    return dice
