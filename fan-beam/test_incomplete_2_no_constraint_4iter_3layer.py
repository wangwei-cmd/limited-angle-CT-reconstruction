import tensorflow as tf
import numpy as np
import os
import incomplete_2_no_constraint_4iter_3layer as net
import matplotlib.pyplot as plt
import h5py
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

data = h5py.File('fan_incompelet_data_max_alpha=18_h=0.05_beta=0_1_150_R=600_256x256_test.mat','r')
f_label, f_ini, sin_label, sin_ini = data['f_label'][:], data['f_ini'][:], data['sin_label'][:], data['sin_incomplete'][:]
del data
f_label, f_ini, sin_label, sin_ini = np.transpose(f_label, [2, 1, 0]), np.transpose(f_ini, [2, 1, 0]), np.transpose(
    sin_label,
    [2, 1,
     0]), np.transpose(
    sin_ini, [2, 1, 0])
f_label, f_ini, sin_label, sin_ini = np.expand_dims(f_label, -1), np.expand_dims(f_ini, -1), np.expand_dims(
    sin_label,
    -1), np.expand_dims(
    sin_ini, -1)

u_img = f_label.astype('float32')
f_img = sin_label.astype('float32')
u_ini_img = f_ini.astype('float32')
f_noisy_img = sin_ini.astype('float32')
del f_label
del f_ini
del sin_label
del sin_ini

M = np.max(np.max(u_ini_img, 1), 1)
M=np.reshape(M, [np.shape(M)[0],1,1,1])
u_img=u_img/M
f_img=f_img/M
u_ini_img =u_ini_img/M
f_noisy_img =f_noisy_img/M

A = h5py.File('A_fan_matrix_256x256__hbeta=1_max_alpha=18_halpha=0.05R=600.mat', 'r')
vx, vy, val, shape = A['vx'][:], A['vy'][:], A['val'][:], A['shap'][:]
vx, vy, val, shape = np.transpose(vx)[0], np.transpose(vy)[0], np.transpose(val)[0], np.transpose(shape)[0]
index = np.stack([vx, vy], 1).astype(np.int64)
shape = shape.astype(np.int64)
val = val.astype('float32')
A = tf.sparse.SparseTensor(index, val, shape)
A = tf.sparse.reorder(A)
del vx
del vy
del val
del index

AT = h5py.File('fan_beam_matrix_256x256__hbeta=1_max_alpha=18_halpha=0.05R=600.mat', 'r')
vx, vy, val, shape, w_c = AT['vx'][:], AT['vy'][:], AT['val'][:], AT['shap'][:], AT['w_c'][:]
vx, vy, val, shape, w_c = np.transpose(vx)[0], np.transpose(vy)[0], np.transpose(val)[0], np.transpose(shape)[0], \
                          np.transpose(w_c)[0]
index = np.stack([vx, vy], 1).astype(np.int64)
shape = shape.astype(np.int64)
val = val.astype('float32')
AT = tf.sparse.SparseTensor(index, val, shape)
AT = tf.sparse.reorder(AT)
del vx
del vy
del val
del index

Ac = h5py.File('Ac_256x256_alpha=-18_0.05_18_beta=0_1_149_R=600.mat', 'r')
vx, vy, val, shape= Ac['vx1'][:], Ac['vy1'][:], Ac['vv1'][:], Ac['shap1'][:]
vx, vy, val, shape= np.transpose(vx)[0], np.transpose(vy)[0], np.transpose(val)[0], np.transpose(shape)[0]

index = np.stack([vx, vy], 1).astype(np.int64)
shape = shape.astype(np.int64)
val = val.astype('float32')
Ac = tf.sparse.SparseTensor(index, val, shape)
Ac = tf.sparse.reorder(Ac)
del vx
del vy
del val
del index

R=600
fx,fy=256,256
assert(np.shape(u_img)[1]==256)
vk = np.ceil(np.sqrt(2) * max(fx, fy))
rvk = np.floor(vk / 2)
max_alpha = np.ceil(np.arcsin(rvk / R)/np.pi*180)
alpha = np.arange(-max_alpha, max_alpha+0.05, 0.05) * np.pi / 180
assert(np.mod(len(alpha),2)==1)
beta = np.arange(0, 150)*np.pi/180
alpha = alpha.astype('float32')
La=len(alpha)
Lb=len(beta)
LL=217
iternum=4
Model = net.Aiteration(iternum, AT, A, Ac, w_c,alpha,Lb,LL, out_size=(256, 256))
ckpt='./ckpt/incomplete_2_no_constraint_4iter_3layer'

L=len(u_ini_img)


_=Model([f_noisy_img[0:1],u_ini_img[0:1]])
Model.load_weights(ckpt)
print('load weights, done')
batch=10
iter = list(range(0, tf.shape(f_noisy_img).numpy()[0], batch))
prediction = np.zeros([tf.shape(f_noisy_img).numpy()[0],256,256,1])
prediction_1 = np.zeros([tf.shape(f_noisy_img).numpy()[0],256,256,1])
for i in range(len(iter)):
    vx=[f_noisy_img[iter[i]:iter[i] + batch],u_ini_img[iter[i]:iter[i] + batch]]
    tmp = Model(vx)
    prediction[iter[i]:iter[i] + batch]=tmp[1].numpy()
    prediction_1[iter[i]:iter[i] + batch]=Model.iradon_fan(tmp[0]).numpy()
    print(i)

ii=np.random.randint(0,L)
print('show figure:',ii)
plt.imshow(u_ini_img[ii,:,:,0],cmap='gray')
plt.figure()
plt.imshow(prediction[ii,:,:,0],cmap='gray')
plt.show()
plt.imshow(prediction_1[ii,:,:,0],cmap='gray')
plt.show()

vy=tf.cast(u_img,tf.float32)
pp=tf.image.psnr(vy,tf.cast(prediction,tf.float32),tf.reduce_max(vy)).numpy()
qq=tf.image.ssim(vy,tf.cast(prediction,tf.float32),tf.reduce_max(vy)).numpy()
print('average psnr:',tf.reduce_mean(pp).numpy())
print('average ssim:',tf.reduce_mean(qq).numpy())
pp1=tf.image.psnr(vy,tf.cast(prediction_1,tf.float32),tf.reduce_max(vy)).numpy()
qq1=tf.image.ssim(vy,tf.cast(prediction_1,tf.float32),tf.reduce_max(vy)).numpy()
print('average psnr:',tf.reduce_mean(pp1).numpy())
print('average ssim:',tf.reduce_mean(qq1).numpy())
plt.imshow(vy[ii,:,:,0],cmap='gray')
plt.show()