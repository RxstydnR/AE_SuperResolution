import keras
import cv2
import glob
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler

import tensorflow as tf


def get_model():
    input_img = tf.keras.layers.Input(shape=(256, 256, 3))

    l1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(input_img)
    l2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l1)
    l3 = tf.keras.layers.MaxPool2D(padding='same')(l2)

    l4 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l3)
    l5 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l4)
    l6 = tf.keras.layers.MaxPool2D(padding='same')(l5)

    l7 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l6)

    l8 = tf.keras.layers.UpSampling2D()(l7)
    l9 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l8)
    l10 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l9)

    l11 = tf.keras.layers.add([l10, l5])

    l12 = tf.keras.layers.UpSampling2D()(l11)
    l13 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l12)
    l14 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l13)

    l15 = tf.keras.layers.add([l14, l2])

    encoder = tf.keras.models.Model(inputs=(input_img), outputs=l7)

    # relu & mse
    decoded_image = tf.keras.layers.Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l15)
    autoencoder = tf.keras.models.Model(inputs=(input_img), outputs=decoded_image)

    # sigmoid & binary cross entropy https://blog.keras.io/building-autoencoders-in-keras.html
    # decoded_image = tf.keras.layers.Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='sigmoid', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l15)
    # autoencoder = tf.keras.models.Model(inputs=(input_img), outputs=decoded_image)
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return (encoder, autoencoder)


# 学習率
def step_decay(epoch):
    x = 0.001
    if epoch >= 30: x = 0.0005
    if epoch >= 60: x = 0.0001
    return x



def get_training_data(images_location):
  
    real_image_treat_as_y = []
    downsize_image_treat_as_x = []
    for img in glob.glob(images_location+"/**/*.jpg", recursive=True)[:1000]:
        try:
            # Original
            image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if np.ndim(image) <=2:
                continue

            width,height,color = image.shape
            if width>=256 and height>=256 and color==3:
                center_w, center_h = width//2, height//2
                image = image[center_w-128:center_w+128,center_h-128:center_h+128]
                real_image_treat_as_y.append(image)
                    
                # reshaped_image = cv2.resize(image, (256, 256))
                # if reshaped_image.shape[-1] == 3:
                #     real_image_treat_as_y.append(reshaped_image)

                # Original → Low → Low Original
                image = cv2.resize(image, (100, 100))
                reshaped_image = cv2.resize(image, (256, 256))
                downsize_image_treat_as_x.append(reshaped_image)

        except:
          pass

    return (np.array(downsize_image_treat_as_x), np.array(real_image_treat_as_y))



parser = argparse.ArgumentParser(prog='train.py')
parser.add_argument('--data_path', type=str, required=True, help='The path of image data.')
parser.add_argument('--save_path', type=str, required=True, help='The path of folder to save results.')
opt = parser.parse_args()


if __name__ == "__main__":

    os.makedirs(opt.save_path,exist_ok=True)

    data_path = opt.data_path
    X,Y = get_training_data(data_path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    Y_train = Y_train.astype('float32') / 255.
    Y_test = Y_test.astype('float32') / 255.

    encoder, autoencoder = get_model()
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Callbacks
    lr_decay = LearningRateScheduler(step_decay)
    
    history = autoencoder.fit(
        X_train, # downized_image
        Y_train, # real_images
        epochs=90,
        batch_size=4,
        shuffle=True,
        callbacks=[lr_decay],
        validation_split=0)


    n_epoch = len(history.history['loss'])
    
    plt.figure()
    plt.plot(range(n_epoch), history.history['loss'], label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend() 
    plt.savefig(opt.save_path+"/SR_autoencoder_history.jpg")
    plt.show(),plt.clf(),plt.close()

    try:
        autoencoder.save(opt.save_path+'/model.h5')
    except:
        print("Cannot save model")

    # autoencoder = keras.models.load_model(opt.save_path+'/model.h5')
    outputs = autoencoder.predict(X_test)

    for i in range(len(X_test)):
        plt.figure(dpi=150)

        plt.subplot(131)
        plt.imshow(X_test[i])
        plt.axis("off")
        plt.title("Low (input)")

        plt.subplot(132)
        plt.imshow(outputs[i])
        plt.axis("off")
        plt.title("Hight (pred)")

        plt.subplot(133)
        plt.imshow(Y_test[i])
        plt.axis("off")
        plt.title("Hight (GT)")

        plt.tight_layout()
        plt.savefig(opt.save_path+f"/{i}.jpg", bbox_inches='tight', pad_inches=0.1)
        plt.show(),plt.clf(),plt.close()




    
    