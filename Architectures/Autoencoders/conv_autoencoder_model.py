from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

    
input_img = Input(shape=(64, 64, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, kernel_size=4, strides=2 activation='relu', padding=(1,1))(input_img) # B,  32, 32, 32
x = Conv2D(32, kernel_size=4, strides=2 activation='relu', padding=(1,1))(x) # B,  32, 16, 16
x = Conv2D(32, kernel_size=4, strides=2 activation='relu', padding=(1,1))(x) # B,  32,  8,  8
x = Conv2D(32, kernel_size=4, strides=2 activation='relu', padding=(1,1))(x) # B,  32,  4,  4
x = x.reshape(-1, 32*4*4) ## B, 512
x = Dense(256, activation='relu')(x) ## 256
encoded = Dense(z_dims*2, activation='relu')(x) ##2*z_dims

# at this point the representation is (z_dims *2)

x = Dense(256, activation='relu')(encoded) ## 256
x = Dense(512, activation='relu')(x) ## 512
x = x.reshape(4, 4, 32)   # B,  32,  4,  4
x = Conv2DTranspose(32, kernel_size=4, strides=2, activation='relu', padding=(1,1))(x) # B,  32,  8,  8
x = Conv2DTranspose(32, kernel_size=4, strides=2, activation='relu', padding=(1,1))(x) # B,  32, 16, 16
x = Conv2DTranspose(32, kernel_size=4, strides=2, activation='relu', padding=(1,1))(x) # B,  32, 32, 32
decoded = Conv2DTranspose(32, kernel_size=4, strides=2, activation='relu', padding=(1,1))(x) # B,  nc, 64, 64

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')