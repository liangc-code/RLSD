# Reinforcement Learning based Denoising Model for Seismic Random Noise Attenuation 
This is the implementation of the [paper](https://ieeexplore.ieee.org/document/10106047) in the IEEE Transactions on Geoscience and Remote Sensing (TGRS).
We provide the sample codes for training and testing.

## Requirements
- Python 3.5+
- Chainer 5.0+
- ChainerRL 0.5+
- Cupy 5.0+
- OpenCV 3.4+

You can install the required libraries by the command `pip install -r requirements.txt`.
We checked this code on cuda-11.4 and cudnn-8.2.2.

## Folders
The folder `code` contains the training and test codes.
`data` is used to store the data for training and testing, you need to put your data in it.

## Usage

### Data preparation
Preparing a four-dimensional data as training set, where the first and second dimensions represent the size of the two-dimensional training patches, the third dimension is the number of training patch, and the fourth dimension is 2, which denotes the clean-noisy data pair. Put the training set in the folder `data\train` and named `train_dataset.mat`.

Put the testing data in the folder `data\test` and named `clean.mat` `input.mat`.

### Training
After completing the preparation of training data, please go to `code` and run `train.py` to train the RLSD model.

### Testing
After completing the training of RLSD model and the preparation of testing data, please go to `code` and run `test.py` to test the RLSD model.

## Note
For reasons of data confidentiality, we only provide the code. All data need to be prepared by yourself.

Pay attention to the first and second stages of the curriculum learning in the paper and prepare the corresponding training set, which is the key to ensure the convergence of the RLSD model.

## References
We use the local similarity proposed by [[Chen+, 2016]](https://library.seg.org/doi/10.1190/geo2014-0227.1) to compute the reward function for fine-tuning stage of the proposed RLSD model. We would thank them and obtained the code from [here](https://github.com/chenyk1990/pyortho).

Our implementation is based on the following articles. We would like to thank them. 
- [PixelRL: Fully Convolutional Network With Reinforcement Learning for Image Processing](https://ieeexplore.ieee.org/document/8936404)

## Citation
If you use our code in your research, please cite our paper.
```
@article{liang2023rlsd,
    title={Reinforcement Learning based Denoising Model for Seismic Random Noise Attenuation},
    author={Liang,Chao and Lin,Hongbo and Ma,Haitao},
    journal={IEEE Transactions on Geoscience and Remote Sensing (TGRS)},
    year={2023},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/TGRS.2023.3268718}
}
```
