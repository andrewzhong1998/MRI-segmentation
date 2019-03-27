import numpy as np
import tensorflow as tf

def convert_to_one_hot(Y, C):
    Y = Y.astype(int)
    Y = np.eye(C)[Y]
    return Y

def Dice(pred,true,labels=[0,1,2,3]):
    l = len(labels)
    ret = np.zeros(l)
    for i in range(l):
        lab = labels[i]
        p = pred==lab
        t = true==lab
        ret[i] = 2*float(np.logical_and(p,t).sum())/(p.sum()+t.sum()+1e-8)
    return ret

def load_npdataset():
    #pretrain_x = np.load("/proj/NIRAL/users/mingzhi/npdata64/pretrain_x_resized.npy")
    #pretrain_y = np.load("/proj/NIRAL/users/mingzhi/npdata64/pretrain_y_resized.npy")
    pretrain_x = np.load("/users/andrew/desktop/npdata64/pretrain_x_resized.npy")
    pretrain_y = np.load("/users/andrew/desktop/npdata64/pretrain_y_resized.npy")
    pretrain_y = convert_to_one_hot(pretrain_y, 4)

    #new_pretrain_x = np.zeros(shape=(22,64,64,64,1))
    #new_pretrain_x[:,:,:,:,0] = pretrain_x[:,:,:,:,0]

    pretrain_x -= np.mean(pretrain_x, axis=0)
    pretrain_x /= np.std(pretrain_x, axis=0) + 1e-8

    return pretrain_x, pretrain_y

def unet(input_size=(64, 64, 64, 2)):
    inputs = tf.keras.layers.Input(shape=input_size)

    conv1 = tf.keras.layers.Conv3D(96, 3, padding="same", activation=tf.nn.relu)(inputs)
    conv1 = tf.keras.layers.Conv3D(96, 3, padding="same", activation=tf.nn.relu)(conv1)
    conv1 = tf.keras.layers.Conv3D(96, 3, padding="same", activation=tf.nn.relu)(conv1)
    pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(conv1)

    conv2 = tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding="same", activation=tf.nn.relu)(pool1)
    conv2 = tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding="same", activation=tf.nn.relu)(conv2)
    pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(conv2)

    conv3 = tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding="same", activation=tf.nn.relu)(pool2)
    pool3 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(conv3)

    donv4 = tf.keras.layers.Conv3D(filters=32, kernel_size=2, padding="same", activation=tf.nn.relu)(
        tf.keras.layers.UpSampling3D(size=(2, 2, 2))(pool3))
    merg4 = tf.keras.layers.concatenate([donv4, conv3])
    conv4 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding="same", activation=tf.nn.relu)(merg4)

    donv5 = tf.keras.layers.Conv3D(filters=32, kernel_size=2, padding="same", activation=tf.nn.relu)(
        tf.keras.layers.UpSampling3D(size=(2, 2, 2))(conv4))
    merg5 = tf.keras.layers.concatenate([donv5, conv2])
    conv5 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding="same", activation=tf.nn.relu)(merg5)
    conv5 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding="same", activation=tf.nn.relu)(conv5)

    donv6 = tf.keras.layers.Conv3D(filters=32, kernel_size=2, padding="same", activation=tf.nn.relu)(
        tf.keras.layers.UpSampling3D(size=(2, 2, 2))(conv5))
    merg6 = tf.keras.layers.concatenate([donv6, conv1])
    conv6 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding="same", activation=tf.nn.relu)(merg6)
    conv6 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding="same", activation=tf.nn.relu)(conv6)
    conv6 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding="same", activation=tf.nn.relu)(conv6)

    outputs = tf.keras.layers.Conv3D(filters=4, kernel_size=3, padding="same", activation=tf.nn.softmax)(conv6)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model

x_train, y_train = load_npdataset()
model = unet()
model.fit(x_train, y_train, epochs=20, batch_size=2)
pred = model.predict(x_train, batch_size=2)
pred = np.argmax(pred, axis=-1)
y_train = np.argmax(y_train, axis=-1)
print(Dice(pred, y_train))
