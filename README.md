

# CMRRWNet

This is the baseline of the MICCAI2026 Challenge: Generalized Analysis of Vessels in Eye Edition 2 (GAVE2)



## Data format

![fig2](https://github.com/user-attachments/assets/7568ccc4-f4d9-4d31-92ca-0a076528ad6f)

* Segmentation: Above is an example of our provided arteriovenous labels. Our code always expects the images to be RGB images with pixel values in the range [0, 255] and the labels to be RGB images with the following segmentation maps in each channel:

    + R: Artery
    + G: Interection of vessels
    + B: Vein

    The masks should be binary images with pixel values in the range [0, 255]. The predictions will be saved in the same format as the masks. To align with the model prediction, in training stage and evaluation stage, our code uses the union of R and G channels (intersection) as the artery label, the union of B and G channels (intersection) as the vein label, and the union of R,G,B channels as the vessel label. This process just for groundtruth, not for prediction result. It can be found in the related code. Our training output is three channels:
    
	+ R: Artery
    + G: Vessel
    + B: Vein

Here's an example of a model prediction, its RGB channels represents artery, vessel and vein respectively.

![fig_example](https://github.com/user-attachments/assets/848e896c-1eb4-4f70-a1a8-c79e9d8aeb03)



## Preparation


The code was tested using Python3.10.12.
However, it should work with other Python versions and package managers.
Just make sure to install the required packages listed in `requirements.txt`. 


### Environment settings
Make sure you have installed conda in advance.
Create and activate Python environment
```
conda create -n cmrrwnet python==3.10
conda activate cmrrwnet
```

Update `pip`.

```sh
pip install --upgrade pip
```

Install requirements using `requirements.txt`.

```sh
pip3 install -r requirements.txt
```


### Preparing Dataset

You can download the GAVE2 dataset through the ["GAVE2 challenge"](https://aistudio.baidu.com/competition/detail/1463/0/introduction) on AI studio. Put the dataset in the `./data`. Before proceeding, please register CFP and FFA images using [MINIMA](https://github.com/LSXI7/MINIMA).The dataset directory structure is following:
```sh
|-data
|	|-training
|	|	|-av        # artery/vein label
|	|	|-images    # color fundus images
|	|	|-masks     # ROI masks
|	|	|-FFA_A     # early_FFA images
|	|	|-FFA_AV    # late_FFA images
|	|-validation
|		|-images    # color fundus images
|		|-masks     # ROI masks
|		|-FFA_A     # early_FFA images
|		|-FFA_AV    # late_FFA images
```


### Project structure

```sh
|-train
|	|-config		# for training
|	|...
|	|-train.py         # step 1
|...
|-get_predictions.py 	# step 2
|-get_biomarker.py		# step 3

```

### Preprocess (optional)

You can preprocess the images offline using the `preprocess.py` script which in the directory `Code/Tool/`. The script will enhance the images and masks and save them in the specified directory.

```bash
python preprocess.py --i <path_to_the_images> --m <path_to_the_images-ROI_mask> --s <path_to_save_enhanced_images>
```



## Run your code

### :one: Training

All training code can be found through the entrance of training script `train.py`, and the configuration file, with all the hyperparameters and command line arguments, is `config.py`.

```bash
python train/train.py
```


### :two: Get predictions

After the model trained, the predictions can be generated using the following command(please modify the configurations first). If you use preprocess for training dataset, pls do the same for test.

```bash
python get_predictions.py
```


### :three: Get Biomarker

You need to first obtain the optic disk segmentation result from [MNet_DeepCDR](https://github.com/HzFu/MNet_DeepCDR).
You can extract vascular biomarkers using the following command (please modify the configurations first).

```bash
python get_biomarker.py 
```





## Contact

If you have any questions or problems with the code or the paper, please do not hesitate to open an issue in this repository (preferred) or contact me at `pengqiyu2004@163.com`.


## Acknowledge
Our project code is built based on the [rrwnet](https://github.com/j-morano/rrwnet) project. The authors' outstanding work, code  and  kind help are gratefully acknowledged.
We also thanks the authors of  ["MNet_DeepCDR"](https://github.com/HzFu/MNet_DeepCDR) ,[MINIMA](https://github.com/LSXI7/MINIMA)

