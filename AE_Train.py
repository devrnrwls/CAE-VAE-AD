# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:23:05 2021

@author: Šimon Bilík

This module provides the autoencoder training based on the folder dataset structure.

Please select the desired model from the module models.py as the model argument

"""

import argparse
import matplotlib.pyplot as plt

from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

from models import build_bae1_model, build_bae2_model, build_mvt_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train convolutional AE and save the model')
    parser.add_argument('--data_path', default='../IndustryBiscuit_KerasApp/', type=str, help='path to dataset')
    parser.add_argument('--model', default='bae1'  , type=int, help='prepared model')
    parser.add_argument('--height', default=256, type=int, help='height of images')
    parser.add_argument('--width', default=256, type=int, help='width of images')
    parser.add_argument('--channel', default=3, type=int, help='channel of images')
    parser.add_argument('--num_epoch', default=500, type=int, help='the number of epochs')
    parser.add_argument('--batch_size', default=20  , type=int, help='mini batch size')

    args = parser.parse_args()

    return args


def main():
    """main function"""
    args = parse_args()

    data_path = args.data_path
    model = args.model
    height = args.height
    width = args.width
    channel = args.channel
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    
    # Change the batchsize according to your system RAM
    train_batchsize = val_batchsize = batch_size
    
    # Path constants
    train_dir = data_path + 'train'
    validation_dir = data_path + 'valid'

    # Set model
    if model == 'bae1':
        autoencoder = build_bae1_model(height, width, channel)
        modelSavePath = './data/bae1/'
    elif model == 'bae2':
        autoencoder = build_bae2_model(height, width, channel)
        modelSavePath = './data/bae2/'
    elif model == 'mvt':
        autoencoder = build_mvt_model(height, width, channel)
        modelSavePath = './data/mvt/'
    else:
        raise Exception('Unknown model!')
    
    # Load the normalized images
    train_datagen = ImageDataGenerator(
        rescale = 1./255) 
        # samplewise_center = True,
        # samplewise_std_normalization = True,
        # rotation_range = 360,
        # width_shift_range = 0.1,
        # height_shift_range = 0.1,
        # horizontal_flip = True)
    
    validation_datagen = ImageDataGenerator(
        rescale = 1./255)
        # samplewise_center = True,
        # samplewise_std_normalization = True)
        
    # Data generator for training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (height, width), 
        batch_size = train_batchsize,
        color_mode = "rgb",
        class_mode = 'input',
        shuffle = True)
    
    # Data generator for validation data
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir, 
        target_size = (height, width), 
        batch_size = val_batchsize,
        color_mode = "rgb",
        class_mode = 'input', 
        shuffle = False)
    
    # Configure the tensorboard callback
    # tbCallBack = tensorflow.keras.callbacks.TensorBoard(
    #     log_dir = './data/', 
    #     histogram_freq = 0,
    #     update_freq = 'batch',
    #     write_graph = True, 
    #     write_images = True)
    
    # Configure the early stopping callback
    esCallBack = callbacks.EarlyStopping(
        monitor = 'loss', 
        patience = 3)
    
    # Show a summary of the model. Check the number of trainable parameters
    autoencoder.summary()
    
    # Configure the model for training
    autoencoder.compile(
        loss = 'mean_squared_error', 
        optimizer = optimizers.Adam()) 
        # metrics = [tensorflow.keras.metrics.AUC(), tensorflow.keras.metrics.Accuracy()])
    
    # Train the model
    trainHistory = autoencoder.fit(
        train_generator, 
        steps_per_epoch = train_generator.samples/train_generator.batch_size, 
        epochs = num_epoch,   
        validation_data = validation_generator, 
        validation_steps = validation_generator.samples/validation_generator.batch_size, 
        verbose = 1,
        callbacks = [esCallBack])
        # callbacks = [tbCallBack, esCallBack])
    
    # Save the model
    autoencoder.save(modelSavePath)
    
    # Plot the history and save the curves
    train_loss = trainHistory.history['loss']
    val_loss = trainHistory.history['val_loss']
    
    fig, axarr = plt.subplots(2)
    tempTitle = " Training and Validation Loss"
    fig.suptitle(tempTitle, fontsize=14, y=1.08)
    
    axarr[0].plot(train_loss)
    axarr[0].set(xlabel = "Number of Epochs", ylabel = "Training Loss [-]")
    
    axarr[1].plot(val_loss)
    axarr[1].set(xlabel = "Number of Epochs", ylabel = "Validation Loss [-]")
    
    fig.tight_layout()
    fig.savefig(modelSavePath + 'losses.png')


if __name__ == '__main__':
    main()
