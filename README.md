# Lightweight Context-Aware Network Using Partial-Channel Transformation for Real-Time Semantic Segmentation
[Lightweight Context-Aware Network Using Partial-Channel Transformation for Real-Time Semantic Segmentation](https://ieeexplore.ieee.org/document/10411824)  

<div align=center><img src="https://github.com/lztjy/LCNet/blob/master/model/cmp.png" height=350 width=600 /></div>
<div align=center><img src="https://github.com/lztjy/LCNet/blob/master/model/LCNet.png"/></div>
<div align=center><img src="https://github.com/lztjy/LCNet/blob/master/model/PCT.png" height=350 width=600/></div>
<div align=center><img src="https://github.com/lztjy/LCNet/blob/master/model/TCA.png"/></div>
<div align=center><img src="https://github.com/lztjy/LCNet/blob/master/model/DAD.png"/></div>
<div align=center><img src="https://github.com/lztjy/LCNet/blob/master/model/DADvis.png"/></div>
<div align=center><img src="https://github.com/lztjy/LCNet/blob/master/model/cityscapes.png"/></div>

# Segmentation performance of LCNet
<table class="tg">
<thead>
  <tr>
    <th class="tg-amwm">S1</th>
    <th class="tg-amwm">S2</th>
    <th class="tg-amwm">Crop Size*</th>
    <th class="tg-amwm">Dataset</th>
    <th class="tg-amwm">Pretrained</th>
    <th class="tg-amwm">Train type</th>
    <th class="tg-amwm">mIoU</th>
    <th class="tg-amwm">Params</th>
    <th class="tg-amwm">Speed</th>
    <th class="tg-amwm">Location</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh">3</td>
    <td class="tg-baqh">7</td>
    <td class="tg-baqh">512,1024</td>
    <td class="tg-baqh">Cityscapes</td>
    <td class="tg-baqh">No</td>
    <td class="tg-baqh">trainval</td>
    <td class="tg-baqh">73.3</td>
    <td class="tg-baqh">0.51</td>
    <td class="tg-baqh">185</td>
    <td class="tg-baqh">-</td>
  </tr>
  <tr>
    <td class="tg-baqh">3</td>
    <td class="tg-baqh">7</td>
    <td class="tg-c3ow">1024,1024</td>
    <td class="tg-c3ow">Cityscapes</td>
    <td class="tg-c3ow">No</td>
    <td class="tg-c3ow">trainval</td>
    <td class="tg-c3ow">73.8</td>
    <td class="tg-baqh">0.51</td>
    <td class="tg-baqh">142</td>
    <td class="tg-baqh">-</td>
  </tr>
  <tr>
    <td class="tg-baqh">3</td>
    <td class="tg-baqh">11</td>
    <td class="tg-c3ow">512,1024</td>
    <td class="tg-c3ow">Cityscapes</td>
    <td class="tg-c3ow">No</td>
    <td class="tg-c3ow">trainval</td>
    <td class="tg-c3ow">74.3</td>
    <td class="tg-baqh">0.74</td>
    <td class="tg-baqh">136</td>
    <td class="tg-baqh">-</td>
  </tr>
  <tr>
    <td class="tg-baqh">3</td>
    <td class="tg-baqh">11</td>
    <td class="tg-c3ow">1024,1024</td>
    <td class="tg-c3ow">Cityscapes</td>
    <td class="tg-c3ow">No</td>
    <td class="tg-c3ow">train</td>
    <td class="tg-c3ow">75.6</td>
    <td class="tg-baqh">0.74</td>
    <td class="tg-baqh">117</td>
    <td class="tg-baqh">-</td>
  </tr>
  <tr>
    <td class="tg-baqh">3</td>
    <td class="tg-baqh">11</td>
    <td class="tg-c3ow">1024,1024</td>
    <td class="tg-c3ow">Cityscapes</td>
    <td class="tg-c3ow">No</td>
    <td class="tg-c3ow">trainval</td>
    <td class="tg-c3ow">75.8</td>
    <td class="tg-baqh">0.74</td>
    <td class="tg-baqh">117</td>
    <td class="tg-baqh">-</td>
  </tr>
</tbody>
</table>

\* Represents the resolution of the input image cropping in the training phase.

# Preparation
You need to download the Cityscapes and CamVid datasets and place the symbolic links or datasets of the Cityscapes and CamVid datasets in the dataset directory. Our file directory is consistent with DABNet (https://github.com/Reagan1311/DABNet).

```
dataset
  ├── camvid
  |    ├── train
  |    ├── test
  |    ├── val 
  |    ├── trainannot
  |    ├── testannot
  |    ├── valannot
  |    ├── camvid_trainval_list.txt
  |    ├── camvid_train_list.txt
  |    ├── camvid_test_list.txt
  |    └── camvid_val_list.txt
  ├── cityscapes
  |    ├── gtCoarse
  |    ├── gtFine
  |    ├── leftImg8bit
  |    ├── cityscapes_trainval_list.txt
  |    ├── cityscapes_train_list.txt
  |    ├── cityscapes_test_list.txt
  |    └── cityscapes_val_list.txt           
```        
# How to run

## 1 Training
### 1.1 Cityscapes
> python train.py 

### 1.2 CamVid
> python train.py --dataset camvid --train_type trainval --max_epochs 1000 --lr 1e-3  --input_size 360,480

## 2 Testing
> python test.py --dataset ${camvid, cityscapes} --checkpoint ${CHECKPOINT_FILE}
### 2.1 Cityscapes  
> python test.py --dataset cityscapes --checkpoint "./checkpoints/LCNet_3_11_1024_train.pth"

To convert the training lables to class lables.
> python trainID2labelID.py
> Package the file into xxx.zip 
> Submit the zip file to https://www.cityscapes-dataset.com/submit/.
> You can get the results from the https://www.cityscapes-dataset.com/submit/.
### 2.2 CamVid
> python test.py --dataset camvid --checkpoint ${CHECKPOINT_FILE}

## 3. fps
> python eval_forward_time.py --size 512,1024

 
 ## Citation
@ARTICLE{  
&emsp;&emsp;10411824,  
&emsp;&emsp;author={Shi, Min and Lin, Shaowen and Yi, Qingming and Weng, Jian and Luo, Aiwen and Zhou, Yicong},  
&emsp;&emsp;journal={IEEE Transactions on Intelligent Transportation Systems},   
&emsp;&emsp;title={Lightweight Context-Aware Network Using Partial-Channel Transformation for Real-Time Semantic Segmentation},   
&emsp;&emsp;year={2024},  
&emsp;&emsp;volume={},  
&emsp;&emsp;number={},  
&emsp;&emsp;pages={1-16}  
}
  
 ## Reference
 
 https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks
 
 https://github.com/Reagan1311/DABNet
 

