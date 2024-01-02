The world runs on a wide variety of neural networks. In this assignment, we play with the mother of them all, feed-forward neural networks (FNN).

# Basic Components 

* Dense Layer: 	a fully connected layer, defined by the dimensions of its input and output.
* ReLU activation Layer
* Dropout Layer: check [here](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html) for details.
* Softmax Layer: check [here](https://d2l.ai/chapter_linear-classification/softmax-regression.html) for details.


# Dataset Description 

We use the EMNIST letters dataset. The dataset contains 28x28 images of letters from the Latin alphabet. The train-validation dataset is splited as 85%-15%.

# Library usage

* Reading the images/possible augmentation: 
* opencv, pillow
* Dataset loading: torchvision
* Visualization: matplotlib, seaborn
* Progress bar: tqdm
* Data manipulation: numpy, pandas
* Model saving and loading: Pickle
* Performance metrics and statistics: scipy, sklearn


## Dependencies
* Python 3.11
* torchvision 
```bash
pip install torchvision
```