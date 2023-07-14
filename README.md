# StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks

This repository provides an implementation of StackGAN, a model for text-to-image synthesis using stacked generative adversarial networks. 
![Framework](examples/framework.jpg)

## Dependencies

- Python 2.7
- TensorFlow 0.12

The following optional dependencies are needed if you want to use additional features:
- Torch (for pre-trained char-CNN-RNN text encoder)
- skip-thought (for skip-thought text encoder)

To install the required packages, add the project folder to your PYTHONPATH and use pip to install the following packages:
- prettytensor
- progressbar
- python-dateutil
- easydict
- pandas
- torchfile

## Dataset

To create the dataset for training, follow these steps:

1. Download the preprocessed char-CNN-RNN text embeddings for birds and flowers and save them to the `Data/` directory. You can download them from the following links:
   - [Birds Text Embeddings](https://drive.google.com/open?id=0B3y_msrWZaXLT1BZdVdycDY5TEE)
   - [Flowers Text Embeddings](https://drive.google.com/open?id=0B3y_msrWZaXLaUc0UXpmcnhaVmM)

2. Download the bird and flower image datasets and extract them to the `Data/birds/` and `Data/flowers/` directories, respectively. You can find the download links in the original paper.

3. Preprocess the images using the provided scripts:
   - For birds: `python misc/preprocess_birds.py`
   - For flowers: `python misc/preprocess_flowers.py`

## Dataset Creation Formula

To create a weighted balanced improved dataset, we can use the following formula:
D_wbi = f(D_org, T_org, w)

Where:
- D_wbi represents the weighted balanced improved dataset.
- f is a function that takes the original dataset D_org, the corresponding text descriptions T_org, and a weight vector w as inputs.
- w is a weight vector that assigns weights to each sample in the dataset.

The function f performs the following steps:
1. Compute the importance scores for each sample in the dataset based on its text description using a text analysis method such as BERT for embedding.
2. Normalize the importance scores to ensure they sum up to 1.
3. Assign weights to each sample in the dataset based on the normalized importance scores.
4. Select samples from the original dataset D_org according to their weights to create the weighted balanced improved dataset D_wbi.

The weighted balanced improved dataset D_wbi aims to have a more balanced distribution of samples across different text descriptions, with higher weights given to samples that are considered more important based on their text descriptions.

Please note that the specific implementation details of the function f and the choice of the weight vector w may vary depending on the specific requirements and characteristics of the dataset.


