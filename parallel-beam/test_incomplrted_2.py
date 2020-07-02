import tensorflow as tf
import numpy as np
import os
import incomplete_2 as net
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

AT = np.load('AT_' + str(180) + '_512x512' + '.npz')
val = AT['name1'].astype('float32')
index = AT['name2']
shape = AT['name3']
AT = tf.sparse.SparseTensor(index, val, shape)
# AT = tf.cast(AT, tf.float32)
A = np.load('A_' + str(180) + '_512x512' + '.npz')
val = A['name1'].astype('float32')
index = A['name2']
shape = A['name3']
A = tf.sparse.SparseTensor(index, val, shape)
del val
del index

udir='./test/limited_angle/'
Model=net.Aiteration(5,AT,A,s_shape = (725, 180),out_size=(512,512))
ckpt='./cpkt/incomplete_2_lambda=0.5'
# ckpt='./512x512/limited-angles/weights/1/incomplete_2_lambda=0.5'
u_ini_img = np.load(udir + 'ini,angle=' + '0-149' + '_no_scale__0.5.npy').astype('float32')
f_img = np.load(udir + '/f,angle=' + str(180) + '_no_scale__0.5.npy').astype('float32')
f_noisy_img = np.zeros(f_img.shape).astype('float32')
L=len(u_ini_img)
f_noisy_img[:, :, 0:149, :] = f_img[:,:,0:149,:]
M = np.max(np.max(u_ini_img, 1), 1)
M=np.reshape(M, [np.shape(M)[0],1,1,1])
u_ini_img =u_ini_img/M
f_noisy_img =f_noisy_img/M
_=Model([f_noisy_img[0:1],u_ini_img[0:1]])
Model.load_weights(ckpt)
print('load weights, done')
batch=10
iter = list(range(0, tf.shape(f_noisy_img).numpy()[0], batch))
prediction = np.zeros([tf.shape(f_noisy_img).numpy()[0],512,512,1])
prediction_1 = np.zeros([tf.shape(f_noisy_img).numpy()[0],512,512,1])
for i in range(len(iter)):
    vx=[f_noisy_img[iter[i]:iter[i] + batch],u_ini_img[iter[i]:iter[i] + batch]]
    tmp = Model(vx)
    prediction[iter[i]:iter[i] + batch]=tmp[1].numpy()
    prediction_1[iter[i]:iter[i] + batch]=Model.iradon(tmp[0]).numpy()
    print(i)

ii=np.random.randint(0,L)
print('show figure:',ii)
plt.imshow(u_ini_img[ii,:,:,0],cmap='gray')
plt.figure()
plt.imshow(prediction[ii,:,:,0],cmap='gray')
plt.show()
plt.imshow(prediction_1[ii,:,:,0],cmap='gray')
plt.show()
u_img = np.load(udir + 'u_CT_img_no_scale.npy').astype('float32')
u_img=u_img/M
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