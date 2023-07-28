# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:23:05 2021

@author: Šimon Bilík

This module defines autoencoder models later used in the AE_train.py script. Feel free to define any new models if necessary.

"""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Basic AE model
def build_bae_model_old(height=128, width=128, channel=3):
    """
    build basic autoencoder model
    """
    input_img = Input(shape=(height, width, channel))

    # Encode-----------------------------------------------------------
    net = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    net = MaxPooling2D((2, 2), padding='same')(net)
    net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
    net = MaxPooling2D((2, 2), padding='same')(net) 
    net = Conv2D(4, (3, 3), activation='relu', padding='same')(net)
    encoded = MaxPooling2D((2, 2), padding='same', name='enc')(net)

    # Decode---------------------------------------------------------------------
    net = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    net = UpSampling2D((2, 2))(net)
    net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
    net = UpSampling2D((2, 2))(net)
    net = Conv2D(16, (3, 3), activation='relu', padding='same')(net)
    net = UpSampling2D((2, 2))(net)
    decoded = Conv2D(channel, (3, 3), activation='sigmoid', padding='same')(net)
    # ---------------------------------------------------------------------

    return Model(input_img, decoded)

# Basic AE model
def build_bae1_model(height=128, width=128, channel=3):
    """
    build basic autoencoder model
    """
    input_img = Input(shape=(height, width, channel))

    # Encode-----------------------------------------------------------
    net = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    net = MaxPooling2D((2, 2), padding='same')(net)
    net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
    net = MaxPooling2D((2, 2), padding='same')(net) 
    net = Conv2D(4, (3, 3), activation='relu', padding='same')(net)
    net = MaxPooling2D((2, 2), padding='same', name='enc')(net)
    net = Dense(10, activation='relu')(net)
    encoded = Dense(1000, activation='relu')(net)

    # Decode---------------------------------------------------------------------
    net = Dense(10, activation='relu')(encoded)
    net = Conv2D(4, (3, 3), activation='relu', padding='same')(net)
    net = UpSampling2D((2, 2))(net)
    net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
    net = UpSampling2D((2, 2))(net)
    net = Conv2D(16, (3, 3), activation='relu', padding='same')(net)
    net = UpSampling2D((2, 2))(net)
    decoded = Conv2D(channel, (3, 3), activation='sigmoid', padding='same')(net)
    # ---------------------------------------------------------------------

    return Model(input_img, decoded)

# Basic AE model
def build_bae2_model(height=128, width=128, channel=3):
    """
    build basic autoencoder model
    """
    input_img = Input(shape=(height, width, channel))

    # Encode-----------------------------------------------------------
    net = Conv2D(8, (5, 5), activation='relu', padding='same')(input_img)
    net = MaxPooling2D((2, 2), padding='same')(net) 
    net = Conv2D(4, (3, 3), activation='relu', padding='same')(net)
    encoded = MaxPooling2D((2, 2), padding='same', name='enc')(net)

    # Decode---------------------------------------------------------------------
    net = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    net = UpSampling2D((2, 2))(net)
    net = Conv2D(8, (5, 5), activation='relu', padding='same')(net)
    net = UpSampling2D((2, 2))(net)
    decoded = Conv2D(channel, (3, 3), activation='sigmoid', padding='same')(net)
    # ---------------------------------------------------------------------

    return Model(input_img, decoded)

# MVTec
def build_mvt_model(height=128, width=128, channel=3):

    input_img = Input(shape=(height, width, channel))  # adapt this if using `channels_first` image data format
    
    # Encode-----------------------------------------------------------
    x = Conv2D(32, (4, 4), strides=2 , activation='relu', padding='same')(input_img)
    x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(128, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
    encoded = Conv2D(1, (8, 8), strides=1, padding='same', name='enc')(x)
    
    # Decode---------------------------------------------------------------------
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(encoded)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = UpSampling2D((4, 4))(x)
    x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(channel, (8, 8), activation='sigmoid', padding='same')(x)
    # ---------------------------------------------------------------------
    
    return Model(input_img, decoded)
