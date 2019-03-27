from keras.models import *
from keras.layers import *
from keras.optimizers import *

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
    pretrain_x = np.load("/proj/NIRAL/users/mingzhi/npdata64/pretrain_x_resized.npy")
    pretrain_y = np.load("/proj/NIRAL/users/mingzhi/npdata64/pretrain_y_resized.npy")
    #pretrain_x = np.load("/users/andrew/desktop/npdata64/pretrain_x_resized.npy")
    #pretrain_y = np.load("/users/andrew/desktop/npdata64/pretrain_y_resized.npy")
    pretrain_y = convert_to_one_hot(pretrain_y, 4)

    #new_pretrain_x = np.zeros(shape=(22,64,64,64,1))
    #new_pretrain_x[:,:,:,:,0] = pretrain_x[:,:,:,:,0]

    pretrain_x -= np.mean(pretrain_x, axis=0)
    pretrain_x /= np.std(pretrain_x, axis=0) + 1e-8

    return pretrain_x, pretrain_y

def unet(input_size=(64, 64, 64, 2)):
    inputs = Input(input_size)

    conv1 = Conv3D(96, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv3D(96, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = Conv3D(96, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = Dropout(0.5)(MaxPooling3D(pool_size=(2, 2, 2))(conv1))

    conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = Dropout(0.5)(MaxPooling3D(pool_size=(2, 2, 2))(conv2))

    conv3 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    pool3 = Dropout(0.5)(MaxPooling3D(pool_size=(2, 2, 2))(conv3))

    up4 = Conv3D(32, 2, activation='relu', padding="same", kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(pool3))
    merge4 = concatenate([conv3, up4], axis=-1)
    conv4 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge4)

    up5 = Conv3D(32, 2, activation='relu', padding="same", kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv4))
    merge5 = concatenate([conv2, up5], axis=-1)
    conv5 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv3D(32, 2, activation='relu', padding="same", kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv5))
    merge6 = concatenate([conv1, up6], axis=-1)
    conv6 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    outputs = Conv3D(4, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(conv6)
    outputs = Reshape(target_shape=(262144,4))(outputs)

    model = Model(input=inputs, output=outputs)
    #model.compile(optimizer=Adam(lr=1e-4), loss="categorical_crossentropy", metrics=['accuracy'])
    weights = np.array((1.0, 1.0, 1.0, 1.0))
    # weights = np.array((1.29891961, 18.17444055, 7.3159837, 26.02806273))
    model.compile(optimizer=Adam(lr=1e-4), loss="categorical_crossentropy", metrics=['accuracy'])
    # model.summary()

    return model

x_train, y_train = load_npdataset()
y_train = y_train.reshape((22,262144,4))
weights = np.array((.1, 2., 1., 4.))
y_train = y_train*weights
model = unet()
model.fit(x_train, y_train, epochs=350, batch_size=2, validation_split=0.1)
pred = model.predict(x_train, batch_size=2)
pred = np.argmax(pred, axis=-1)
y_train = np.argmax(y_train, axis=-1)
print(Dice(pred, y_train))
