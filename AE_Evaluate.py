# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:23:05 2021

@author: Šimon Bilík

This script evaluates the trained CAE model obtained at script AE_Train.py

"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the trained CAE')

    parser.add_argument('--data_path', default='../IndustryBiscuit_KerasApp/', type=str, help='path to dataset')
    parser.add_argument('--docs_path', default='./docs/', type=str, help='path to docs')
    parser.add_argument('--labels_path', default='./data/cookie_labels.npz', type=str, help='path to labels')
    parser.add_argument('--action', default='Train', type=int, help='Process training, or testing data. Set as Train, or Test')
    parser.add_argument('--height', default=256, type=int, help='height of images')
    parser.add_argument('--width', default=256, type=int, help='width of images')
    parser.add_argument('--batch_size', default=5  , type=int, help='mini batch size')

    args = parser.parse_args()

    return args

def load_data(data_to_path):
    """load data
    data should be compressed in npz
    """
    data = np.load(data_to_path)

    try:
        train_label = data['train']
        valid_label = data['valid']
        test_label = data['test']
    except:
        print('Loading data should be numpy array and has "images" and "labels" keys.')
        sys.exit(1)

    return train_label, valid_label, test_label

def flat_feature(enc_out):
    """flat feature of CAE encoded layer
    """
    enc_out_flat = []

    s1, s2, s3 = enc_out[0].shape
    s = s1 * s2 * s3
    for con in enc_out:
        enc_out_flat.append(con.reshape((s,)))

    return np.array(enc_out_flat)

def encode_data(data_generator, model_path, labels, output_path, reduce = False):
    """Returns and saves the encoded data"""
    
    # Load the model
    autoencoder = load_model(model_path)
    
    # Get the encoded data
    layer_name = 'enc'
    encoded_layer = Model(inputs = autoencoder.input, outputs = autoencoder.get_layer(layer_name).output)
    enc_out = encoded_layer.predict(data_generator)
    
    # Flat and reduce the data if set
    if reduce:
        enc_out = flat_feature(enc_out)
        pca = PCA(n_components=50)
        enc_out = pca.fit_transform(enc_out)
    
    np.savez(output_path, ae_out = enc_out, labels = labels)
    
    return enc_out
    
def decode_data(data_generator, model_path, labels, output_path, reduce = False):
    """Returns and saves the decoded data"""
    
    # Load the model
    autoencoder = load_model(model_path)
    
    # Get the decoded data
    aec_out = autoencoder.predict(data_generator)
    
    # Flat and reduce the data if set
    if reduce:
        aec_out = flat_feature(aec_out)
        pca = PCA(n_components=50)
        aec_out = pca.fit_transform(aec_out)
    
    np.savez(output_path, ae_out = aec_out, labels = labels)
    
    return aec_out

def get_average(input_data):
    """ Computes the average picture"""
    
    average = np.mean(input_data, 0)
    plt.imshow(average)
    
    return average

def NormalizeData(data):
    """Performs the data normalization"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def main():
    """main function"""
    args = parse_args()

    action = args.action
    data_path = args.data_path
    docs_path = args.docs_path
    labels_path = args.labels_path

    height = args.height
    width = args.width
    test_batchsize = args.batch_size

    # Set the action
    if action == 'Train':
        train = True
    elif action == 'Test':
        train = False
    else:
        raise Exception('Variable action was not set correctly, please set it as Train or Test...')
    
    # Path constants
    train_dir = data_path + 'train'
    test_dir = data_path + 'test'
    
    train_datagen = ImageDataGenerator(
        rescale = 1./255)
        # samplewise_center = True,
        # samplewise_std_normalization = True)
    
    test_datagen = ImageDataGenerator(
        rescale = 1./255)
        # samplewise_center = True,
        # samplewise_std_normalization = True)
    
    # Data generator for train data
    train_generator = train_datagen.flow_from_directory(
        train_dir, 
        target_size = (height, width), 
        batch_size = test_batchsize,
        color_mode = "rgb",
        class_mode = 'input',
        shuffle = False)
    
    # Data generator for test data
    test_generator = test_datagen.flow_from_directory(
        test_dir, 
        target_size = (height, width), 
        batch_size = test_batchsize,
        color_mode = "rgb",
        class_mode = 'input',
        shuffle = False)
    
    # Load all labels
    train_label, _, test_label = load_data(labels_path)
    
    if train:
        # Get the encodeded data
        bae1_enc = NormalizeData(encode_data(train_generator, "./data/bae1/", train_label, "./data/bae1/bae1_enc_train.npz"))
        bae2_enc = NormalizeData(encode_data(train_generator, "./data/bae2/", train_label, "./data/bae2/bae2_enc_train.npz"))
        mvt_enc = NormalizeData(encode_data(train_generator, "./data/mvt/", train_label, "./data/mvt/mvt_enc_train.npz"))
        
        # Get the decoded data
        bae1_aec = NormalizeData(decode_data(train_generator, "./data/bae1/", train_label, "./data/bae1/bae1_dec_train.npz"))
        bae2_aec = NormalizeData(decode_data(train_generator, "./data/bae2/", train_label, "./data/bae2/bae2_dec_train.npz"))
        mvt_aec = NormalizeData(decode_data(train_generator, "./data/mvt/", train_label, "./data/mvt/mvt_dec_train.npz"))
        
        # get_average(bae1_aec)
        
        # Get the original images
        img_nc, _ = train_generator._get_batches_of_transformed_samples(np.array([0]))
        img_nc = img_nc[0, :, :, :]
        
        img_cd, _ = train_generator._get_batches_of_transformed_samples(np.array([27]))
        img_cd = img_cd[0, :, :, :]
        
        img_so, _ = train_generator._get_batches_of_transformed_samples(np.array([35]))
        img_so = img_so[0, :, :, :]
        
        img_nm, _ = train_generator._get_batches_of_transformed_samples(np.array([200]))
        img_nm = img_nm[0, :, :, :]
    else:
        # Get the encodeded data
        bae1_enc = NormalizeData(encode_data(test_generator, "./data/bae1/", test_label, "./data/bae1/bae1_enc_test.npz"))
        bae2_enc = NormalizeData(encode_data(test_generator, "./data/bae2/", test_label, "./data/bae2/bae2_enc_test.npz"))
        mvt_enc = NormalizeData(encode_data(test_generator, "./data/mvt/", test_label, "./data/mvt/mvt_enc_test.npz"))
        
        # Get the decoded data
        bae1_aec = NormalizeData(decode_data(test_generator, "./data/bae1/", test_label, "./data/bae1/bae1_dec_test.npz"))
        bae2_aec = NormalizeData(decode_data(test_generator, "./data/bae2/", test_label, "./data/bae2/bae2_dec_test.npz"))
        mvt_aec = NormalizeData(decode_data(test_generator, "./data/mvt/", test_label, "./data/mvt/mvt_dec_test.npz"))
    
        # Get the original images
        img_nc, _ = test_generator._get_batches_of_transformed_samples(np.array([0]))
        img_nc = img_nc[0, :, :, :]     
        
        img_cd, _ = test_generator._get_batches_of_transformed_samples(np.array([27]))
        img_cd = img_cd[0, :, :, :]
        
        img_so, _ = test_generator._get_batches_of_transformed_samples(np.array([35]))
        img_so = img_so[0, :, :, :]
        
        img_nm, _ = test_generator._get_batches_of_transformed_samples(np.array([200]))
        img_nm = img_nm[0, :, :, :]
    
    # Plot the encoded samples from all classes
    fig1, axarr = plt.subplots(4,4)
    fig1.suptitle('Encoded images')
    
    axarr[0,0].set_title("Original")
    axarr[0,0].imshow(img_nc)
    axarr[0,0].axis('off')
    axarr[1,0].imshow(img_cd)
    axarr[1,0].axis('off')
    axarr[2,0].imshow(img_so)
    axarr[2,0].axis('off')
    axarr[3,0].imshow(img_nm)
    axarr[3,0].axis('off')
    
    axarr[0,1].set_title("BAE1")
    axarr[0,1].imshow(bae1_enc[0])
    axarr[0,1].axis('off')
    axarr[1,1].imshow(bae1_enc[27])
    axarr[1,1].axis('off')
    axarr[2,1].imshow(bae1_enc[35])
    axarr[2,1].axis('off')
    axarr[3,1].imshow(bae1_enc[200])
    axarr[3,1].axis('off')
    
    axarr[0,2].set_title("BAE2")
    axarr[0,2].imshow(bae2_enc[0])
    axarr[0,2].axis('off')
    axarr[1,2].imshow(bae2_enc[27])
    axarr[1,2].axis('off')
    axarr[2,2].imshow(bae2_enc[35])
    axarr[2,2].axis('off')
    axarr[3,2].imshow(bae2_enc[200])
    axarr[3,2].axis('off')
    
    axarr[0,3].set_title("MVT")
    axarr[0,3].imshow(mvt_enc[0])
    axarr[0,3].axis('off')
    axarr[1,3].imshow(mvt_enc[27])
    axarr[1,3].axis('off')
    axarr[2,3].imshow(mvt_enc[35])
    axarr[2,3].axis('off')
    axarr[3,3].imshow(mvt_enc[200])
    axarr[3,3].axis('off')
    
    plt.tight_layout()
    plt.show()

    fig1.savefig(docs_path + 'AE_Overview_Encoded_' + action + '.png')
    
    # Plot the decoded samples from all classes
    fig2, axarr = plt.subplots(4,4)
    fig2.suptitle('Decoded images')
    
    axarr[0,0].set_title("Original")
    axarr[0,0].imshow(img_nc)
    axarr[0,0].axis('off')
    axarr[1,0].imshow(img_cd)
    axarr[1,0].axis('off')
    axarr[2,0].imshow(img_so)
    axarr[2,0].axis('off')
    axarr[3,0].imshow(img_nm)
    axarr[3,0].axis('off')
    
    axarr[0,1].set_title("BAE1")
    axarr[0,1].imshow(bae1_aec[0])
    axarr[0,1].axis('off')
    axarr[1,1].imshow(bae1_aec[27])
    axarr[1,1].axis('off')
    axarr[2,1].imshow(bae1_aec[35])
    axarr[2,1].axis('off')
    axarr[3,1].imshow(bae1_aec[200])
    axarr[3,1].axis('off')
    
    axarr[0,2].set_title("BAE2")
    axarr[0,2].imshow(bae2_aec[0])
    axarr[0,2].axis('off')
    axarr[1,2].imshow(bae2_aec[27])
    axarr[1,2].axis('off')
    axarr[2,2].imshow(bae2_aec[35])
    axarr[2,2].axis('off')
    axarr[3,2].imshow(bae2_aec[200])
    axarr[3,2].axis('off')
    
    axarr[0,3].set_title("MVT")
    axarr[0,3].imshow(mvt_aec[0])
    axarr[0,3].axis('off')
    axarr[1,3].imshow(mvt_aec[27])
    axarr[1,3].axis('off')
    axarr[2,3].imshow(mvt_aec[35])
    axarr[2,3].axis('off')
    axarr[3,3].imshow(mvt_aec[200])
    axarr[3,3].axis('off')
    
    plt.tight_layout()
    plt.show()

    fig2.savefig(docs_path + 'AE_Overview_Decoded_' + action + '.png')


if __name__ == '__main__':
    main()
