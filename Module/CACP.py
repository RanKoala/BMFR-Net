# -*- coding: utf-8 -*-

from keras.backend import expand_dims
from tensorflow.python.keras.layers import BatchNormalization, UpSampling2D, GlobalAveragePooling2D
from tensorflow.python.keras.layers import Conv2D, Activation, Add


def SSCAC(input, _kenrel_size, return_filter_num, stride, _dilation_rate_list, _name):
    x = Conv2D(return_filter_num, _kenrel_size, padding='same', activation=None, strides=stride,
               dilation_rate=_dilation_rate_list[0], name=_name + '_1')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(return_filter_num, _kenrel_size, padding='same', activation=None, strides=stride,
               dilation_rate=_dilation_rate_list[1], name=_name + '_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(return_filter_num, _kenrel_size, padding='same', activation=None, strides=stride,
               dilation_rate=_dilation_rate_list[2], name=_name + '_3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def CACP(input, filter):
    reduceBlock = Conv2D(filters=256, kernel_size=(1, 1), strides=1, activation=None, padding='valid',
                         kernel_initializer='he_normal',
                         name="reduceFeature")(input)

    pooling_block = GlobalAveragePooling2D(name="GlobalAveragePooling2D")(reduceBlock)
    pooling_block = expand_dims(expand_dims(pooling_block, 1), 1)
    pooling_block = Conv2D(filters=256, kernel_size=(1, 1), strides=1, activation="relu", padding='valid',
                           kernel_initializer='he_normal',
                           name="pooling_block_conv1x1")(pooling_block)
    pooling_block = BatchNormalization()(pooling_block)
    pooling_block = UpSampling2D(size=(32, 32), name="upSample_block", interpolation='bilinear')(pooling_block)

    atrous_block123 = SSCAC(reduceBlock, (3, 3), 256, 1, [1, 2, 3], "atrous_123")  # k=3,5,7  R=3,7,13
    atrous_block135 = SSCAC(reduceBlock, (3, 3), 256, 1, [1, 3, 5], "atrous_135")  # k=3,7,11 R=3,9,19
    atrous_block139 = SSCAC(reduceBlock, (3, 3), 256, 1, [1, 3, 9], "atrous_139")  # k=3,7,19 R=3,9,27

    total_layers = Add()([reduceBlock, pooling_block, atrous_block123, atrous_block135, atrous_block139])
    result_ASPP = Conv2D(filters=filter, kernel_size=(1, 1), strides=1, activation=None, padding='valid',
                         kernel_initializer='he_normal',
                         name="result_ASPP")(total_layers)
    return result_ASPP
