**Warning:** This repository is outdated and it won't be maintained in future. Please refer to the repository https://github.com/boortel/AE-Reconstruction-And-Feature-Based-AD and to the ModelClassificationErrM.py module for its enhanced reimplemantation.

# Feature space reduction as data preprocessing for the anomaly detection

This repository implements the data preprocessing methods for the anomaly detection described in the paper [Feature space reduction as data preprocessing for the anomaly detection](https://arxiv.org/abs/2203.06747) and it is partialy based on the [Unsupervised Anomaly detection with One/Class Support Vector Machine](https://github.com/hiram64/ocsvm-anomaly-detection) repository.

#### Dependencies
Install the TensorFlow 2 via pip: https://www.tensorflow.org/install

Install the remaining requirements using:
```
pip install -r requirements.txt
```

## How to use
### 1. Prepare data
Prepare data and labels to use. For instance, CIFAR10 is composed of 10 classes and each label should express unique class and be integer. Industrial cookie consists of four classes (one OK and three anomalous), but for the anomaly detection task only OK and common NOK classes are considered. These prepared data should be placed in the data directory.

You can download CIFAR10 data via :  
https://www.kaggle.com/janzenliu/cifar-10-batches-py

Industrial cookie dataset (more suitable for the anomaly detection) is available at:
https://www.kaggle.com/datasets/imonbilk/industry-biscuit-cookie-dataset

Put them in "data" directory and run the following code to compress them into NPZ file.
```
python makeCifar10Npz.py

or

python makeCustomNpz.py
```
After running this code, you can get *npz* data under "data" directory.


#### (Optional)
When you use your own dataset, please prepare npz file as the same format as CIFAR-10. Customize the *makeCustomNpz.py* script if necessary.
```
data = np.load('your_data.npz')
data.files
-> ['images', 'labels'] # "images" and "labels" keys'

data['labels']
-> array([6, 9, 9, ..., 5, 1, 7]) # labels is the vector composed of integers which correspond to each class identifier.

Note : Please be careful fo input image size of model.py.
You might need to change network architecture's parameter so that it can deal with your images.
```


### 2. Train CAE
Three CAE models (BAE1, BAE2 and MVTec) are defined in the module *models.py*. Train and save the desired model by running the script
```
python AE_train.py
```
Model is going to be saved in the *.pb* file format in the folder *data*.


### 3. Evaluate CAE
Evaluate the trained models using the saved *.pb* data by running the script
```
python AE_evaluate.py
```
to obtain encoded and decoded images of all three defined models in the corresponding *data* subfolders.


### 4. Obtain reconstruction error metrics
Get the reconstruction error metrics by running the script
```
python AD_ErrorMetrics.py
```

### 5. Run Anomaly Detection
**TODO**

## References

Please cite following paper in your further work:

```
@inproceedings{BUT171163,
  author="Šimon {Bilík}",
  title="Feature space reduction as data preprocessing for the anomaly detection",
  address="Brno University of Technology, Faculty of Electrical Engineering",
  booktitle="Proceedings I of the 27th Conference STUDENT EEICT 2021",
  chapter="171163",
  howpublished="online",
  institution="Brno University of Technology, Faculty of Electrical Engineering",
  year="2021",
  month="april",
  pages="415--419",
  publisher="Brno University of Technology, Faculty of Electrical Engineering",
  type="conference paper"
}
```
