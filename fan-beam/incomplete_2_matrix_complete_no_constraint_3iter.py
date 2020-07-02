import tensorflow as tf
import numpy as np
import datetime
from scipy.fftpack import fft2, ifft2, fftfreq, fftshift,dct,idct
import matplotlib.pyplot as plt
import os
import h5py

class Oneiterate(tf.keras.layers.Layer):
    def __init__(self, A, s_shape=(360, 721),out_size=(256,256)):
        super(Oneiterate, self).__init__()
        self.A=A
        self.s_shape=s_shape
        self.out_size = out_size

        # self.lambda1=tf.Variable(initial_value=1.0, trainable=True, name='lambda1')
        # self.lambda2 = tf.Variable(initial_value=1.0, trainable=True, name='lambda2')
        self.sinLayer = []
        self.ctLayer = []
        self.sinLayer.append(tf.keras.layers.Conv2D(64, 5, padding='same', name='sinconv0', activation=tf.nn.relu))
        for layers in range(1, 2 + 1):
            self.sinLayer.append(tf.keras.layers.Conv2D(64, 5, padding='same', name='sinconv%d' % layers, use_bias=False))
            self.sinLayer.append(tf.keras.layers.BatchNormalization(name='sinBN%d' % layers))
            self.sinLayer.append(tf.keras.layers.ReLU(name='sinReLU%d' % layers))
        self.sinLayer.append(tf.keras.layers.Conv2D(1, 5, name='sinconvend', padding='same'))
        ###CTLayer###
        self.ctLayer.append(tf.keras.layers.Conv2D(64, 5, padding='same', name='ctconv0', activation=tf.nn.relu))
        for layers in range(1, 2 + 1):
            self.ctLayer.append(tf.keras.layers.Conv2D(64, 5, padding='same', name='ctconv%d' % layers, use_bias=False))
            self.ctLayer.append(tf.keras.layers.BatchNormalization(name='ctBN%d' % layers))
            self.ctLayer.append(tf.keras.layers.ReLU(name='ctReLU%d' % layers))
        self.ctLayer.append(tf.keras.layers.Conv2D(1, 5, name='ctconvend', padding='same'))

    def radon_fan(self, f):
        batch = tf.shape(f)[0]
        f = tf.reshape(f, [batch, -1])
        f = tf.transpose(f)
        sin = tf.sparse.sparse_dense_matmul(self.A, f)
        sin = tf.reshape(sin, [self.s_shape[0], self.s_shape[1], 1, batch])
        sin = tf.transpose(sin, [3, 0, 1, 2])
        return sin

    def call(self, inputs):
        batch=tf.shape(inputs)[0]
        sin=self.radon_fan(inputs)
        sin=tf.cast(sin,tf.complex64)
        fft=tf.signal.fft2d(sin)
        fft=tf.concat([tf.math.real(fft),tf.math.imag(fft)],axis=0)
        z=self.sinLayer[0](fft)
        for i in range(1,len(self.sinLayer)):
            z=self.sinLayer[i](z)
        z=fft+z
        de_fft=tf.cast(z[0:batch],tf.complex64)+tf.cast(z[batch::],tf.complex64)*1j
        z=tf.signal.ifft2d(de_fft)
        z=tf.cast(tf.math.real(z),tf.float32)
        # z_u=self.iradon(z)

        u=self.ctLayer[0](inputs)
        for i in range(1,len(self.ctLayer)):
            u=self.ctLayer[i](u)
        u=inputs+u

        # f=(tf.signal.idct(z,norm='ortho')+u)
        return [z,u]

class Aiteration(tf.keras.Model):
    def __init__(self,iternum,AT,A,A_c,w_c,alpha,Lb,LL,out_size=(256,256)):
        super(Aiteration, self).__init__()
        self.alpha=alpha
        self.La=len(alpha)
        self.s_shape = [360,self.La]
        # h_beta=beta[1]-beta[0]
        # full_beta=np.pi+2*alpha(-1)
        # LL = np.ceil((full_beta-beta(0)) / h_beta + 1)
        self.Lb = Lb
        self.LL=max(LL,self.Lb)

        self.A_c=A_c
        self.A=A
        self.AT = AT
        self.w_c = w_c
        self.out_size=out_size
        self.lambda1 = tf.Variable(initial_value=tf.constant(0.1,shape=[iternum]), trainable=True, shape=[iternum],name='lambda1')
        self.lambda2 = tf.Variable(initial_value=tf.constant(0.1,shape=[iternum]), trainable=True, shape=[iternum],name='lambda2')
        self.lambda3 = tf.Variable(initial_value=tf.constant(0.1, shape=[iternum]), trainable=True, shape=[iternum],name='lambda3')
        self.lambda4 = tf.Variable(initial_value=tf.constant(1.0, shape=[iternum]), trainable=True, shape=[iternum],name='lambda4')
        self.oneiterstack=[]
        for i in range(iternum):
            self.oneiterstack.append(Oneiterate(A,self.s_shape,out_size))

    def iradon_fan(self,sin_fan):
        cos_alpha = tf.math.cos(self.alpha)
        s_fan_shape = tf.shape(sin_fan)
        h=self.alpha[1]-self.alpha[0]
        sin_fan1 = h * tf.expand_dims(sin_fan[:, :, :, 0] * cos_alpha, -1)
        sin_fan1 = tf.reshape(sin_fan1, [-1, s_fan_shape[2], 1])
        filter_s_fan = tf.nn.conv1d(sin_fan1, tf.cast(tf.reshape(self.w_c, [s_fan_shape[2], 1, 1]), tf.float32), stride=1,
                                    padding='SAME')
        # filter_s_fan1=tf.reshape(filter_s_fan,s_fan_shape)
        filter_s_fan2 = tf.reshape(filter_s_fan, [s_fan_shape[0], -1])
        filter_s_fan2 = tf.transpose(filter_s_fan2)
        rf = tf.sparse.sparse_dense_matmul(self.AT, filter_s_fan2)
        rf = tf.transpose(rf)
        rf = tf.reshape(rf, [s_fan_shape[0], self.out_size[0], self.out_size[1], 1])
        return rf

    def radon_fan(self, f):
        batch = tf.shape(f)[0]
        f = tf.reshape(f, [batch, -1])
        f = tf.transpose(f)
        sin = tf.sparse.sparse_dense_matmul(self.A, f)
        sin = tf.reshape(sin, [self.s_shape[0], self.s_shape[1], 1, batch])
        sin = tf.transpose(sin, [3, 0, 1, 2])
        return sin

    def complete(self, sin):
        shape = tf.shape(sin)
        g1 = tf.reshape(sin, [shape[0], -1])
        g1 = tf.transpose(g1)
        gg = tf.sparse.sparse_dense_matmul(self.A_c, g1)
        gg = tf.transpose(gg)
        gg = tf.reshape(gg, [shape[0], 360 - self.LL, self.La, 1])
        g2 = tf.concat([sin, tf.zeros([shape[0], self.LL - self.Lb, shape[2], 1]), gg], axis=1)
        return g2

    def __call__(self, inputs,training=0):
        [z,u]=self.oneiterstack[0](inputs[1])
        g=inputs[0]
        g=self.complete(g)
        f=inputs[1]
        Af=self.radon_fan(f)
        a1=Af-g
        # a1=tf.concat([a1[:,:,0:self.Lb,:],tf.zeros([shape[0],shape[1],self.LL-self.Lb,1])],axis=2)
        a1=a1[:, 0:self.Lb, :, :]
        a1=self.complete(a1)
        a2=Af-z
        a3=self.iradon_fan(self.lambda1[0]*a1+self.lambda2[0]*a2)
        f = self.lambda4[0]*f-a3+self.lambda3[0]*u
        for i in range(1,len(self.oneiterstack)-1):
            [z,u]=self.oneiterstack[i](f)
            Af = self.radon_fan(f)
            a1 = Af - g
            a1 = a1[:, 0:self.Lb, :, :]
            a1 = self.complete(a1)
            a2 = Af - z
            a3 = self.iradon_fan(self.lambda1[0] * a1 + self.lambda2[0] * a2)
            f = self.lambda4[0]*f-a3+self.lambda3[0]*u
        return [z,u]

def make_model(iternum,AT,A,s_shape = (360, 1601),out_size=(256,256)):
    forward=Aiteration(iternum,AT,A,s_shape,out_size)
    inputs = tf.keras.Input(shape=(None, None, 1))
    output=forward(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def u_function(s):
    u = np.zeros(s.shape)
    index_1 = np.where(s == 0)[0]
    u[index_1] = 1 / 2
    index = np.where(s != 0)[0]
    v = s[index]
    u[index] = (np.cos(v) - 1) / (v ** 2) + np.sin(v) / v
    return u

def w_bfunction(b, s):
    return u_function(b * s) * (b ** 2) / (4 * np.pi ** 2)

@tf.function
def train_step(inputs, model, labels, Loss, Metric, optimizer,vx,vy,epochnum):
    # if epochnum<1000:
    #     weights = 0.9999
    # else:
    #     weights = 0.0001
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=1)
        loss = Loss(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # m1 = Metric(labels, inputs)
    m1 = Metric([labels[0][:,0:150,:,:],labels[1]], inputs)
    m2 = Metric(labels, model(inputs, training=0))
    m3 = Metric(vy, model(vx, training=0))
    return loss, m1, m2, m3


def loss_2(x, y,weights=0.5):
    x0 = tf.cast(x[0], tf.float32)
    x1 = tf.cast(x[1], tf.float32)
    y0 = tf.cast(y[0], tf.float32)
    y1 = tf.cast(y[1], tf.float32)
    shape = tf.cast(tf.shape(x[0]), tf.float32)
    shape1 = tf.cast(tf.shape(x[1]), tf.float32)
    return weights*tf.reduce_sum(tf.math.square(x0 - y0)) / shape[0] / shape[1] / shape[2] / shape[3]\
    +(1-weights)*tf.reduce_sum(tf.math.square(x1 - y1))/shape1[0] / shape1[1] / shape1[2] / shape1[3]


def psnr_2(x, y,max_val=255):
    x0 = tf.cast(x[0], tf.float32)
    x1 = tf.cast(x[1], tf.float32)
    y0 = tf.cast(y[0], tf.float32)
    y1 = tf.cast(y[1], tf.float32)
    batch = tf.cast(tf.shape(x[1])[0], tf.float32)
    psnr1=tf.reduce_sum(tf.image.psnr(x0, y0, max_val=tf.reduce_max(x0))) / batch######psnr of f and de_sin
    psnr3 = tf.reduce_sum(tf.image.psnr(x1, y1, max_val=tf.reduce_max(x1))) / batch#####psnr of u and reconstructed
    return [psnr1,psnr3]


def train(epoch, udir,batch, theta, iternum,  ckpt, restore=0):
    data = h5py.File('fan_incompelet_data_max_alpha=18_h=0.05_beta=0_1_150_R=600_256x256_train.mat','r')
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



    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
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
    Model = Aiteration(iternum, AT, A, Ac, w_c,alpha,Lb,LL, out_size=(256, 256))
    tf.keras.backend.clear_session()


    N=tf.shape(u_img)[0]
    vx=[f_noisy_img[N-5:N],u_ini_img[N-5:N]]
    # vy = u_img[N - 5:N]
    vy=[f_img[N-5:N],u_img[N-5:N]]
    train_data = tf.data.Dataset.from_tensor_slices((u_img[0:N-5],f_img[0:N-5], u_ini_img[0:N-5],f_noisy_img[0:N-5])).shuffle(tf.cast(N-5,tf.int64)).batch(batch)
    # _ = Model(vx[0:1])
    if restore == 1:
        # call the build function in the layers since do not use tf.keras.Input
        ##maybe move the functions in build function to _ini_ need not do this
        _=Model(vx[0:1])
        Model.load_weights(ckpt)
        print('load weights, done')
    for i in range(epoch):
        for iter, ufini in enumerate(train_data):
            u,f, u_ini,f_noisy = ufini
            Loss, m1, m2, m3 = train_step([f_noisy,u_ini], Model, [f,u], loss_2, psnr_2, optimizer, vx, vy, epochnum=i)
            print(iter, "/", i, ":", Loss.numpy(),
                  "psnr_f_fnoisy:", [m1[0].numpy(),m1[1].numpy()],
                  "psnr1", [m2[0].numpy(), m2[1].numpy()],
                  ###psnr of f and f_noisy, u and fbp, u and reconstructe,respectively
                  'psnr3:', [m3[0].numpy(), m3[1].numpy()]
                  )
        if i%2==0:
            Model.save_weights(ckpt)
    Model.save_weights(ckpt)
    # tf.keras.utils.plot_model(Model, 'multi_input_and_output_model.png', show_shapes=True)
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    iternum = 3
    epoch =100

    theta=0
    batch = 2
    udir = "./limited_angle/"
    vdir = "validate"
    train(epoch, udir, batch, theta, iternum, restore=0, ckpt='./512x512/limited-angles/weights/fanbeam/incomplete_2_matrix_compelte_no_constraint_3iter')