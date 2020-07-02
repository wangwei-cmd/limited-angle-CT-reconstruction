import tensorflow as tf
import numpy as np
import datetime
from scipy.fftpack import fft2, ifft2, fftfreq, fftshift,dct,idct
import matplotlib.pyplot as plt
import os

class Oneiterate(tf.keras.layers.Layer):
    def __init__(self, AT, A, s_shape=(725, 180),out_size=(512,512)):
        super(Oneiterate, self).__init__()
        self.A=A
        # self.AT = AT
        self.s_shape=s_shape
        self.out_size = out_size
        w_b = w_bfunction(np.pi,np.linspace(-np.floor(s_shape[0] / 2), s_shape[0] - np.floor(s_shape[0] / 2) - 1, s_shape[0]))
        self.w_b = w_b.astype('float32')

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

    def radon(self, f):
        batch = tf.shape(f)[0]
        f = tf.reshape(f, [batch, -1])
        f = tf.transpose(f)
        sin = tf.sparse.sparse_dense_matmul(self.A, f)
        sin = tf.reshape(sin, [self.s_shape[1], self.s_shape[0], 1, batch])
        sin = tf.transpose(sin, [3, 1, 0, 2])
        return sin

    def call(self, inputs):
        batch=tf.shape(inputs)[0]
        sin=self.radon(inputs)
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
    def __init__(self,iternum,AT,A,s_shape = (725, 180),out_size=(512,512)):
        super(Aiteration, self).__init__()
        self.s_shape = s_shape
        self.A=A
        self.AT = AT
        w_b = w_bfunction(np.pi, np.linspace(-np.floor(s_shape[0] / 2), s_shape[0] - np.floor(s_shape[0] / 2) - 1, s_shape[0]))
        self.w_b = w_b.astype('float32')
        self.lambda1 = tf.Variable(initial_value=tf.constant(0.1,shape=[iternum]), trainable=True, shape=[iternum],name='lambda1')
        self.lambda2 = tf.Variable(initial_value=tf.constant(0.1,shape=[iternum]), trainable=True, shape=[iternum],name='lambda2')
        self.lambda3 = tf.Variable(initial_value=tf.constant(0.1, shape=[iternum]), trainable=True, shape=[iternum],name='lambda2')
        self.lambda4 = tf.Variable(initial_value=tf.constant(1.0, shape=[iternum]), trainable=True, shape=[iternum],name='lambda2')
        self.oneiterstack=[]
        for i in range(iternum):
            self.oneiterstack.append(Oneiterate(AT,A,s_shape,out_size))

    def iradon(self,sin_fan):
        AT, w_b=self.AT,self.w_b
        sin_fan=tf.transpose(sin_fan,perm=[0,2,1,3])
        batch=tf.shape(sin_fan)[0]
        sin_fan1 = tf.reshape(sin_fan, [-1, self.s_shape[0], 1])
        filter_s_fan = tf.nn.conv1d(sin_fan1, tf.expand_dims(tf.expand_dims(w_b,-1),-1), stride=1, padding='SAME')
        filter_s_fan2 = tf.reshape(filter_s_fan, [batch, -1])
        filter_s_fan2 = tf.transpose(filter_s_fan2)
        rf = tf.sparse.sparse_dense_matmul(AT, filter_s_fan2)
        rf = tf.transpose(rf)
        rf = tf.reshape(rf, [batch, 512, 512, 1])
        return 4*rf

    def radon(self, f):
        batch = tf.shape(f)[0]
        f = tf.reshape(f, [batch, -1])
        f = tf.transpose(f)
        sin = tf.sparse.sparse_dense_matmul(self.A, f)
        sin = tf.reshape(sin, [self.s_shape[1], self.s_shape[0], 1, batch])
        sin = tf.transpose(sin, [3, 1, 0, 2])
        return sin

    def __call__(self, inputs,training=0):
        [z,u]=self.oneiterstack[0](inputs[1])
        g=inputs[0]
        shape=tf.shape(inputs[0])
        f=inputs[1]
        Af=self.radon(f)
        a1=Af-g
        a1=tf.concat([a1[:,:,0:149,:],tf.zeros([shape[0],shape[1],31,1])],axis=2)
        a2=Af-z
        a3=self.iradon(self.lambda1[0]*a1+self.lambda2[0]*a2)
        f = self.lambda4[0]*f-a3+self.lambda3[0]*u
        for i in range(1,len(self.oneiterstack)-1):
            [z,u]=self.oneiterstack[i](f)
            Af = self.radon(f)
            a1 = Af - g
            a1 = tf.concat([a1[:, :, 0:149, :], tf.zeros([shape[0], shape[1], 31, 1])], axis=2)
            a2 = Af - z
            a3 = self.iradon(self.lambda1[0] * a1 + self.lambda2[0] * a2)
            f = self.lambda4[0] * f - a3 + self.lambda3[0] * u
        return [z,u]

def make_model(iternum,AT,A,s_shape = (725, 180),out_size=(512,512)):
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
    m1 = Metric(labels, inputs)
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
    angles = np.shape(theta)[0]
    s_shape = (725, 180)
    out_size = (512, 512)
    u_img = np.load(udir + 'u_CT_img_no_scale.npy').astype('float32')
    f_img = np.load(udir + '/f,angle=' + str(180) + '_no_scale__0.5.npy').astype('float32')
    u_ini_img = np.load(udir + 'ini,angle='+'0-149'+'_no_scale__0.5.npy').astype('float32')
    # tmp = np.load(udir + 'f_noisy,angle=' + '0-149' + '_no_scale__0.5.npy').astype('float32')
    f_noisy_img = np.zeros(f_img.shape).astype('float32')
    f_noisy_img[:, :, 0:149, :] = f_img[:,:,0:149,:]

    M = np.max(np.max(u_ini_img, 1), 1)
    M=np.reshape(M, [np.shape(M)[0],1,1,1])
    u_img=u_img/M
    f_img=f_img/M
    u_ini_img =u_ini_img/M
    f_noisy_img =f_noisy_img/M


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
    print('shape of u_img:', u_img.shape)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Model=make_model(iternum,AT,A,s_shape = (725, 180),out_size=(512,512))
    Model=Aiteration(iternum,AT,A,s_shape = (725, 180),out_size=(512,512))
    # Model=Aiteration(2,AT,A,s_shape = (725, 180),out_size=(512,512))
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    iternum = 5
    epoch =100

    batch = 2
    angles = 60
    theta = np.linspace(0, 180, angles, endpoint=False)
    udir = "./limited_angle/"
    vdir = "validate"
    train(epoch, udir, batch, theta, iternum, restore=0, ckpt='./512x512/limited-angles/weights/incomplete_2_lambda=0.5')