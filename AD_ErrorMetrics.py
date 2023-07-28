# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:23:05 2021

@author: Šimon Bilík

This script evaluates the L2 and SSIM based reconstruction error metrics and t-SNE feature reduction

"""

import argparse
import numpy as np
import sklearn.metrics as mt
import matplotlib.pyplot as plt

from PIL import Image
from SSIM_PIL import compare_ssim
from sklearn.manifold import TSNE

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the trained CAE')

    parser.add_argument('--docs_path', default='./docs/', type=str, help='path to docs')
    parser.add_argument('--action', default='Train', type=int, help='Process training, or testing data. Set as Train, or Test')

    args = parser.parse_args()

    return args

def getFeatures_old(inputData, inputAvg):
    """ Computes the feature vectors from the input data"""
    elements = inputData.shape[0]
    ax = None
    
    tr_ssim = []
    tr_mse = []
    
    input_dim = inputData.shape[3]
    
    for sample in range(elements):
        t_ssim = []
        t_mse = []
         
        for dim in range(input_dim):
            
            # Compute the MSE metric
            mse = mt.mean_squared_error(inputAvg[:, :, dim], inputData[sample, :, :, dim])
            
            # Compute the SSIM metric
            avgIm = Image.fromarray(inputAvg[:, :, dim], 'L')
            comImg = Image.fromarray(inputData[sample, :, :, dim], 'L')
            
            ssim = compare_ssim(avgIm, comImg)
            
            t_ssim.append(ssim)
            t_mse.append(mse)
                       
        tr_ssim.append(t_ssim)
        tr_mse.append(t_mse)
        
    tr_ssim = np.array(tr_ssim)
    tr_mse = np.array(tr_mse)
    
    featureMatrix = np.append(tr_ssim, tr_mse, 1)
    
    return featureMatrix

def NormalizeData(data):
    """Performs the data normalization"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def getFeatures(inputData, inputAll):
    """ Computes the feature vectors from the input data"""
    elements = inputData.shape[0]
    ax = None
    
    tr_ssim = []
    tr_L2 = []
    
    # inputData = NormalizeData(inputData)
    # inputAll = NormalizeData(inputAll)
    
    for sample in range(elements):
        
        # Compute the SSIM metric
        avgIm = Image.fromarray(inputAll[sample], 'RGB')
        comImg = Image.fromarray(inputData[sample], 'RGB')
            
        ssim = compare_ssim(avgIm, comImg)
        tr_ssim.append(ssim)
        
        # Compute the L2 normtemp
        L2 = np.sum(np.square(np.subtract(inputAll[sample], inputData[sample])), axis=None)
        
        if L2 < 0:
            print("problem")
        
        tr_L2.append(L2)
        
    tr_ssim = np.array(tr_ssim)
    tr_L2 = np.array(tr_L2)
    
    # tr_ssim = NormalizeData(tr_ssim)
    # tr_L2 = NormalizeData(tr_L2)
    
    featureMatrix = np.column_stack((tr_ssim, tr_L2))
    
    return featureMatrix

def main():
    """main function"""
    args = parse_args()

    action = args.action
    docs_path = args.docs_path

    # Set the action
    if action == 'Train':
        train = True
    elif action == 'Test':
        train = False
    else:
        raise Exception('Variable action was not set correctly, please set it as Train or Test...')

    # Load .npz saved data   
    bae1_train = np.load('./data/bae1/bae1_dec_train.npz')
    bae2_train = np.load('./data/bae2/bae2_dec_train.npz')
    mvt_train = np.load('./data/mvt/mvt_dec_train.npz')
    
    train_all = np.load('./data/cookie_train.npz')
    
    bae1_test = np.load('./data/bae1/bae1_dec_test.npz')
    bae2_test = np.load('./data/bae2/bae2_dec_test.npz')
    mvt_test = np.load('./data/mvt/mvt_dec_test.npz')
    
    test_all = np.load('./data/cookie_test.npz')
    
    try:
        
        train_labels = train_all['labels']
        test_labels = test_all['labels']
        
        train_all = train_all['images']
        test_all = test_all['images']
        
        bae1_train = bae1_train['ae_out']
        bae2_train = bae2_train['ae_out']
        mvt_train = mvt_train['ae_out']
        
        bae1_test = bae1_test['ae_out']
        bae2_test = bae2_test['ae_out']
        mvt_test = mvt_test['ae_out']
    except:
        raise Exception('Loading data should be numpy array and has "ae_out" and "labels" keys.')
    
    # Compute the reconstruction error metrics  
    if train:
        tr_feature_bae1 = getFeatures(bae1_train, train_all)
        tr_feature_bae2 = getFeatures(bae2_train, train_all)
        tr_feature_mvt = getFeatures(mvt_train, train_all)
        
        # Perform the t-SNE and the feature space visualisation
        tr_tsne_bae1 = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=100).fit_transform(tr_feature_bae1)
        tr_tsne_bae2 = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=100).fit_transform(tr_feature_bae2)
        tr_tsne_mvt = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=100).fit_transform(tr_feature_mvt)
        
        # Save the data
        np.savez("./data/bae1/bae1_tr", ae_out = tr_feature_bae1, labels = train_labels)
        np.savez("./data/bae2/bae2_tr", ae_out = tr_feature_bae2, labels = train_labels)
        np.savez("./data/mvt/mvt_tr", ae_out = tr_feature_mvt, labels = train_labels)
        
        np.savez("./data/bae1/bae1_trsn", ae_out = tr_tsne_bae1, labels = train_labels)
        np.savez("./data/bae2/bae2_trsn", ae_out = tr_tsne_bae2, labels = train_labels)
        np.savez("./data/mvt/mvt_trsn", ae_out = tr_tsne_mvt, labels = train_labels)
        
        # Visualise the data
        fig1, axs = plt.subplots(1, 3)
        fig1.suptitle('Train samples')
        axs[0].scatter(tr_tsne_bae1[:, 0], tr_tsne_bae1[:, 1])
        axs[0].set_title("BAE1")
        axs[1].scatter(tr_tsne_bae2[:, 0], tr_tsne_bae2[:, 1])
        axs[1].set_title("BAE2")
        axs[2].scatter(tr_tsne_mvt[:, 0], tr_tsne_mvt[:, 1])
        axs[2].set_title("MVT")
        fig1.legend(["OK"])
        
        plt.tight_layout()
        plt.show()

        fig1.savefig(docs_path + 'trainSamples_TSNE.png')
        
        fig3, axs = plt.subplots(1, 3)
        fig3.suptitle('Train samples')
        axs[0].scatter(tr_feature_bae1[:, 0], tr_tsne_bae1[:, 1])
        axs[0].set_title("BAE1")
        axs[1].scatter(tr_feature_bae2[:, 0], tr_tsne_bae2[:, 1])
        axs[1].set_title("BAE2")
        axs[2].scatter(tr_feature_mvt[:, 0], tr_tsne_mvt[:, 1])
        axs[2].set_title("MVT")
        fig3.legend(["OK"])
        
        plt.tight_layout()
        plt.show()

        fig3.savefig(docs_path + 'trainSamples_L2_SSIM.png')
        
    else:
        ts_feature_bae1 = getFeatures(bae1_test, test_all)
        ts_feature_bae2 = getFeatures(bae2_test, test_all)
        ts_feature_mvt = getFeatures(mvt_test, test_all)
        
        # Perform the t-SNE and the feature space visualisation
        ts_tsne_bae1 = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=100).fit_transform(ts_feature_bae1)
        ts_tsne_bae2 = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=100).fit_transform(ts_feature_bae2)
        ts_tsne_mvt = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=100).fit_transform(ts_feature_mvt)
        
        # Save the data
        np.savez("./data/bae1_ts", ae_out = ts_feature_bae1, labels = test_labels)
        np.savez("./data/bae2_ts", ae_out = ts_feature_bae2, labels = test_labels)
        np.savez("./data/mvt_ts", ae_out = ts_feature_mvt, labels = test_labels)
        
        np.savez("./data/bae1_tssn", ae_out = ts_tsne_bae1, labels = test_labels)
        np.savez("./data/bae2_tssn", ae_out = ts_tsne_bae2, labels = test_labels)
        np.savez("./data/mvt_tssn", ae_out = ts_tsne_mvt, labels = test_labels)
        
        # Visualise the data
        
        fig2, axs = plt.subplots(1, 3)
        fig2.suptitle('Test samples')
        axs[0].scatter(ts_tsne_bae1[199:, 0], ts_tsne_bae1[199:, 1])
        axs[0].scatter(ts_tsne_bae1[:199, 0], ts_tsne_bae1[:199, 1])
        axs[0].set_title("BAE1")
        axs[1].scatter(ts_tsne_bae2[199:, 0], ts_tsne_bae2[199:, 1])
        axs[1].scatter(ts_tsne_bae2[:199, 0], ts_tsne_bae2[:199, 1])
        axs[1].set_title("BAE2")
        axs[2].scatter(ts_tsne_mvt[199:, 0], ts_tsne_mvt[199:, 1])
        axs[2].scatter(ts_tsne_mvt[:199, 0], ts_tsne_mvt[:199, 1])
        axs[2].set_title("MVT")
        fig2.legend(["OK", "NOK"])
        
        plt.tight_layout()
        plt.show()

        fig2.savefig(docs_path + 'testSamples_TSNE.png')
        
        fig4, axs = plt.subplots(1, 3)
        fig4.suptitle('Test samples')
        axs[0].scatter(ts_feature_bae1[199:, 0], ts_feature_bae1[199:, 1])
        axs[0].scatter(ts_feature_bae1[:199, 0], ts_feature_bae1[:199, 1])
        axs[0].set_title("BAE1")
        axs[1].scatter(ts_feature_bae2[199:, 0], ts_feature_bae2[199:, 1])
        axs[1].scatter(ts_feature_bae2[:199, 0], ts_feature_bae2[:199, 1])
        axs[1].set_title("BAE2")
        axs[2].scatter(ts_feature_mvt[199:, 0], ts_feature_mvt[199:, 1])
        axs[2].scatter(ts_feature_mvt[:199, 0], ts_feature_mvt[:199, 1])
        axs[2].set_title("MVT")
        fig4.legend(["OK", "NOK"])
        
        plt.tight_layout()
        plt.show()

        fig4.savefig(docs_path + 'testSamples_L2_SSIM.png')
        
if __name__ == '__main__':
    main()
