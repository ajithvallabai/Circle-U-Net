## Unet 

### Dataset - 

- Camvid
- ICG - TUGRAZ dataset - http://dronedataset.icg.tugraz.at 
- Look Into Person Dataset
- Coco https://github.com/giddyyupp/coco-minitrain

### Installations:


tensorflow-gpu==2.1.4

### Dataset

```
---dataset
          |
          |---data--
                    |
                    |
                    |
                    |------icg_drone
                    |               |
                    |               |-----train_frames--
                    |               |                   |-----train------
                    |               |                   |                |001.jpg
                    |               |                   |                |002.jpg
                    |               |                   |                |003.jpg
                    |               |-----train_masks---
                    |               |                   |-----train------
                    |               |                   |                |001.jpg
                    |               |                   |                |002.jpg
                    |               |                   |                |003.jpg
                    |               |
                    |               |-----val_frames----
                    |               |                   |-----val--------
                    |               |                   |                |055.jpg
                    |               |                   |                |056.jpg
                    |               |                   |                |057.jpg
                    |               |-----val_masks-----
                    |               |                   |-----val--------
                    |               |                   |                |055.jpg
                    |               |                   |                |056.jpg
                    |               |                   |                |057.jpg
                    |
                    |---label_color.txt

```

In ICG semantic drone dataset ,

- "train_frames" could be taken from - semantic_drone_dataset_semantics_v1.1\semantic_drone_dataset\training_set\images

- "train_masks" could be taken from - 
semantic_drone_dataset_semantics_v1.1\semantic_drone_dataset\training_set\gt\semantic\label_images

Please randomly seperate train and val set as you like make sure there are 360 for training and 40 for testing

**Train frames**

all other images except in val set

**Train masks**

all other images except in val set

**Val frames** 

3,19,53,71,89,104,122,139,182,177,216,225,244,263,290,304,320,332,367,386
412,421,438,476,489,507,524,545,567,583,584,585,586,587,588,590,591,592,593,593,594


**Val masks** 

3,19,53,71,89,104,122,139,182,177,216,225,244,263,290,304,320,332,367,386
412,421,438,476,489,507,524,545,567,583,584,585,586,587,588,590,591,592,593,593,594


### Running programs

**Training**

Experiments - Feb 2021

**Attenation Unet**


``` python drone_main.py -d "camvid_small" -idir "dataset/icg_drone/data/" -m "att_unet" -ht 256 -w 256 -bs 5 --loss tversky --num_epochs 60 ```


**Resnet101 Unet**

```python drone_main.py -d "camvid_small" -idir "dataset/icg_drone/data/" -m "res_unet" -ht 256 -w 256 -bs 5 --loss tversky --num_epochs 60 ```

**Unet**

```python drone_main.py -d "camvid_small" -idir "dataset/icg_drone/data/" -m "tiny_unet" -ht 256 -w 256 -bs 5 --loss tversky --num_epochs 60```

**Circlenet** - Tversky loss

```python drone_main.py -d "camvid_small" -idir "dataset/icg_drone/data/" -m "circlenet" -ht 256 -w 256 -bs 5 --loss tversky --num_epochs 60```

**Circlenet** - Categorical cross entropy

```python drone_main.py -d "camvid_small" -idir "dataset/icg_drone/data/" -m "circlenet" -ht 256 -w 256 -bs 5 --loss CCE --num_epochs 60```

**Circlenet with attention**  - Tversky loss

```python drone_main.py -d "camvid_small" -idir "dataset/icg_drone/data/" -m "circle_att_101" -ht 256 -w 256 -bs 5 --loss tversky --num_epochs 60 ```

**Circlenet with attention**  - Categorical cross entropy

```python drone_main.py -d "camvid_small" -idir "dataset/icg_drone/data/" -m "circle_att_101" -ht 256 -w 256 -bs 5 --loss CCE --num_epochs 60 ```

**Attention unet** - Categorical cross entropy
```
python drone_main.py -d "camvid_small" -idir "dataset/icg_drone/data/" -m att_unet -ht 256 -w 256 -bs 5 --loss CCE --num_epochs 60
```

**Resunet** - Categorical cross entropy
```
python drone_main.py -d "camvid_small" -idir "dataset/icg_drone/data/" -m "res_unet" -ht 256 -w 256 -bs 5 --loss CCE --num_epochs 60
```

**Squeezeunet** -CCE
```
python drone_main.py -d "camvid_small" -idir "dataset/icg_drone/data/" -m new_squeezenet  -ht 256 -w 256 -bs 5 --loss CCE --num_epochs 60
```

**Tables to be filled***

Model Name 	        |   loss function |val mIOU   |val accuracy | epochs
-------             | --------        | --------  | -------     |  --------- |
Unet                |	trversky      |           |           |            |
Attenation UNet     |	trversky	  |      	  |           |             |
squeeze-Unet        |	trversky      |	          |	          |             |
Att Sequeeze-Unet   |	trversky      |	          |	          |             |
Resnet101 Unet      |   trversky      |           |           |     60      |
Circle net          |  trversky       |           |           |     60      |
Circle net          |  categorical CE |           |           |     60      |
Circlenet with atten|  trversky       |           |           |     60      |
Circlenet with atten|  categorical CE |           |           |     60      |


**Evaluating model and predicting images**
    
    python evaluate.py -d "camvid" -idir "dataset/camvid/data/" -mt "squeeze_unet_keras" -m "camvid_model_5_epochs.h5" -ht 256 -w 256

    

### UNet papers - SOTA :

- [Growth of Unet](https://paperswithcode.com/method/u-net)

- [Unet - root](https://arxiv.org/pdf/1505.04597.pdf)

Last two years SOTA papers
- [UNet](https://arxiv.org/pdf/1505.04597.pdf)
- [UNet++](https://arxiv.org/pdf/1807.10165.pdf)
- [Att_UNet](https://arxiv.org/pdf/1804.03999.pdf)
- [ResUNet](https://arxiv.org/pdf/1512.03385.pdf)
- [RexUnet](https://arxiv.org/pdf/1611.05431.pdf)
- [Adversarial Learning](https://arxiv.org/pdf/1802.07934.pdf) 
- [NAS-Unet](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8681706)


CVPR 
- [Eff-UNet](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w22/Baheti_Eff-UNet_A_Novel_Architecture_for_Semantic_Segmentation_in_Unstructured_Environment_CVPRW_2020_paper.pdf)
- [Feedback U-net](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w57/Shibuya_Feedback_U-Net_for_Cell_Image_Segmentation_CVPRW_2020_paper.pdf)
- [Enhanced rotation equivariant Unet](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVMI/Chidester_Enhanced_Rotation-Equivariant_U-Net_for_Nuclear_Segmentation_CVPRW_2019_paper.pdf)

ICCV 
- [Reccurent Unets](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Recurrent_U-Net_for_Resource-Constrained_Segmentation_ICCV_2019_paper.html)
- [BCDU-Net](http://openaccess.thecvf.com/content_ICCVW_2019/papers/VRMI/Azad_Bi-Directional_ConvLSTM_U-Net_with_Densley_Connected_Convolutions_ICCVW_2019_paper.pdf) - https://github.com/rezazad68/BCDU-Net




- [GAN based_on UNet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Schonfeld_A_U-Net_Based_Discriminator_for_Generative_Adversarial_Networks_CVPR_2020_paper.pdf)
- [RUNet for super resolution](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WiCV/Hu_RUNet_A_Robust_UNet_Architecture_for_Image_Super-Resolution_CVPRW_2019_paper.pdf)

### Accuracy to be achieved:

- CamVid -

mIOU - 0.70 is target

![assets/squeezenet_accuracy.PNG](SqueezeNet)


- Drone dataset - 

Any accuracy is fine

- Look into Person dataset -

model	|overall acc.	|mean acc.	|mean IoU|
------- | --------      | --------  | -------|
resnet50|	0.792	|0.552	|0.463|
resnet101|	0.805	|0.579	|0.489|
densenet121|	0.826|	0.606|	0.519|
squeezenet|	0.786|	0.543|	0.450|

**Reference for LIP** : https://github.com/hyk1996/Single-Human-Parsing-LIP

**Reference for models:**

[Blog 1](https://medium.com/@sunnerli/simple-introduction-about-hourglass-like-model-11ee7c30138)

### Todo

- Make all available UNets in this repo 
- Create it like a library where you can install and infer on 3 datasets 
