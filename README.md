## GREW-BENCHMARCH

This repository contains the code for our ICCV 2021 paper `Gait Recognition in the Wild: A Benchmark` 


GREW-BENCHMARK project is mainly maintained By Xianda Guo and Zheng Zhu. 

For all main contributors, please check [contributing](#contributing).


## Dataset
Download [GREW](https://www.grew-benchmark.org/download.html) datasets.

You can decompress the compressed package with the following command
```Shell
unzip -P password train.zip (password 是要解压的密码 )
tar -xzvf train.tgz
cd train
ls *.tgz | xargs -n1 tar xzvf
```

Folder structure after decompression is as follows:
```Shell
├── make_pose(Silhouettes, Gait Energy Images (GEIs) , 2D and 3D poses)
    ├── train
        ├── 00001
            ├── 4XPn5Z28
                ├── 00001.png
                ├── 00001_2d_pose.txt
                ├── 00001_3d_pose.txt
            ├── 4XPn5Z28_gei.png
    ├── test
        ├── gallery
            ├── 00001
                ├── 79XJefi8
                    ├── 00001.png
                    ├── 00001_2d_pose.txt
                    ├── 00001_3d_pose.txt
                ├── 79XJefi8_gei.png
        ├── probe
            ├── 01DdvEHX
                ├── 00001.png
                ├── 00001_2d_pose.txt
                ├── 00001_3d_pose.txt
            ├── 01DdvEHX_gei.png
    ├── distractors
        ├── edXJefi8
            ├── 00001.pn
            ├── 00001_2d_pose.txt
            ├── 00001_3d_pose.txt
        ├── edXJefi8_gei.png
        
├── flow(optical flow)
    ├── train
        ├── 00001
            ├── 4XPn5Z28
                ├── 00001_00002_flow.png
    ├── test
        ├── gallery
            ├── 00001
                ├── 79XJefi8
                    ├── 00001_00002_flow.png
        ├── probe
            ├── 01DdvEHX
                ├── 00001_00002_flow.png
    ├── distractors
        ├── edXJefi8
            ├── 00001_00002_flow.png
```
**!!! ATTENTION !!! ATTENTION !!! ATTENTION !!!**

The Silhouettes and Gait Energy Images (GEIs) has been resized to 64*64. And 2D and 3D poses are obtained from the original image without changing the shape.

The header of *_2d_pose.txt is as follows:
```
orl_image_h,orl_image_w,nose_x,nose_y,nose_conf,left_eye_x,left_eye_y,left_eye_conf,right_eye_x,right_eye_y,right_eye_conf,left_ear_x,left_ear_y,left_ear_conf,right_ear_x,right_ear_y,right_ear_conf,left_shoulder_x,left_shoulder_y,left_shoulder_conf,right_shoulder_x,right_shoulder_y,right_shoulder_conf,left_elbow_x,left_elbow_y,left_elbow_conf,right_elbow_x,right_elbow_y,right_elbow_conf,left_wrist_x,left_wrist_y,left_wrist_conf,right_wrist_x,right_wrist_y,right_wrist_conf,left_hip_x,left_hip_y,left_hip_conf,right_hip_x,right_hip_y,right_hip_conf,left_knee_x,left_knee_y,left_knee_conf,right_knee_x,right_knee_y,right_knee_conf,left_ankle_x,left_ankle_y,left_ankle_conf,right_ankle_x,right_ankle_y,right_ankle_conf
```
The header of *_3d_pose.txt is as follows:
```
orl_image_h,orl_image_w,head_x,head_y,head_z,chin_x,chin_y,chin_z,right_shoulder_x,right_shoulder_y,right_shoulder_z,right_elbow_x,right_elbow_y,right_elbow_z,right_wrist_x,right_wrist_y,right_wrist_z,left_shoulder_x,left_shoulder_y,left_shoulder_z,left_elbow_x,left_elbow_y,left_elbow_z,left_wrist_x,left_wrist_y,left_wrist_z,right_hip_x,right_hip_y,right_hip_z,right_knee_x,right_knee_y,right_knee_z,right_ankle_x,right_ankle_y,right_ankle_z,left_hip_x,left_hip_y,left_hip_z,left_knee_x,left_knee_y,left_knee_z,left_ankle_x,left_ankle_y,left_ankle_z
```

## Requirements

### Environment

1. Python 3.6.*
2. CUDA 11.1
3. PyTorch 
4. TorchVision 

### Install
Create a  virtual environment and activate it.
```shell
conda create -n grew python=3.6.12
conda activate grew
```
The code has been tested with PyTorch 1.0 and Cuda 11.1.
```shell
conda install pytorch=1.0.0 torchvision=0.2.1
conda install matplotlib tqdm
conda install tensorboard tensorboardX
conda install scipy scikit-image opencv
```
### Train
Train a model by
```bash
python train.py
```
- `--cache` if set as TRUE all the training data will be loaded at once before the training start.
This will accelerate the training.
**Note that** if this arg is set as FALSE, samples will NOT be kept in the memory
even they have been used in the former iterations. #Default: TRUE

### Test
```bash
python build_submission.py
```
Participants must package the submission.csv for submission using zip xxx.zip $CSV_PATH and then upload it to [codalab](https://competitions.codalab.org/competitions/35463).


## Acknowledgements

Part of the code is adopted from previous works: [GaitSet](https://github.com/AbnerHqC/GaitSet), We thank the original authors for their awesome repos.

## License
The GREW dataset is freely available for free non-commercial use, and may be redistributed under these conditions.
For commercial queries, contact [Zheng Zhu](zhengzhu@ieee.org).


### Citing
If you find this code useful, please consider to cite our work.

```
@inproceedings{zhu2021gait,
    title={Gait Recognition in the Wild: A Benchmark},
    author={Zheng Zhu, Xianda Guo, Tian Yang, Junjie Huang, 
        Jiankang Deng, Guan Huang, Dalong Du,Jiwen Lu, Jie Zhou},
    booktitle={IEEE International Conference on Computer Vision (ICCV)},
    year={2021}              
}
```
### Contributing
Main contributors:

- Xianda Guo, ``xianda_guo@163.com``
