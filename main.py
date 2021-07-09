# from model.fcn import fcn_8s

import time
import tensorflow as tf
import tensorflow.python.keras.backend as backend
from data import *
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from plot_acc_loss import a_l_plot
from Module.BMFRNet import BMFRNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.
session = tf.Session(config=config)
backend.get_session(session)

data_gen_args = dict(rotation_range=20,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     fill_mode='nearest')

Train_Batch_Size = 6
Val_Batach_Size = 6
Epoches = 200
Train_Setp_Per_Epoches = 732
Val_Setp_Per_Epoches = 24
MODEL_TYPE = 'BMFRNet-Net' +'_'+ str(Epoches) + '_' + 'batchSize='+str(Train_Batch_Size)+'_LR=1e-4_'

trainGene = trainGenerator(Train_Batch_Size, 'data/基础数据/Mass256/train', 'image256', 'label256', aug_dict=data_gen_args, save_to_dir=None)
valGen = trainGenerator(Val_Batach_Size, 'data/基础数据/Mass256/validation', 'image256', 'label256', aug_dict=data_gen_args, save_to_dir=None)

model = BMFRNet()
type_model = 'BMFRNet'

filepath = 'checkpoint/BMFRNet.hdf5'
outTestDir = 'BMFRNet'


if os.path.isfile(filepath):
    model.load_weights(filepath)
callbacks_list = [
    ModelCheckpoint(
        filepath=filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True
    ),
    TensorBoard(
        log_dir='E:/unet_log'
    )
]

starttime = time.time()
history = model.fit_generator(generator=trainGene, steps_per_epoch=Train_Setp_Per_Epoches, epochs=Epoches,
                              validation_steps=Val_Setp_Per_Epoches, callbacks=callbacks_list, validation_data=valGen)
endtime = time.time()  # 1*12=12  *5=60
dtime = endtime - starttime
print("Training time：%.8s s" % dtime)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
a_l_plot(acc, val_acc, loss, val_loss,
         MODEL_TYPE,
         r'E:\Ran\oldCode\acc_loss',Epoches,
         ('b', 'Train accuracy'),
         ('r', 'Validation accuracy'),
         ('b', 'Train loss'),
         ('r', 'Validation loss'))
file_handle = open(r'E:\Ran\oldCode\TrainTime.txt', mode='a', encoding='utf-8')
file_handle.write('*******' + MODEL_TYPE + '*******')
now_time = time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='y', m='m', d='d', h='h', M='Min', s='s')
file_handle.write(now_time + '\n')
file_handle.write("Training time：%.8s s" % dtime)
file_handle.writelines('\n')
file_handle.flush()
file_handle.close()


testGene = testGenerator(r"D:\DeepLearning\unet-master\data\gao", num_image=1)
results = model.predict_generator(testGene, 1, verbose=1)
saveResult(outTestDir + '_Epoch_' + str(Epoches), save_path1=r"D:\DeepLearning\unet-master\data\gao",
           save_path2=r'D:\DeepLearning\unet-master\data\gao' + outTestDir + '/', npyfile=results)
model.save_weights(filepath)