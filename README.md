# NanoTrack 

![network](./image/nanotrack_network.png)

- NanoTrack is a lightweight and high speed tracking network which mainly referring to SiamBAN and LightTrack. It is suitable for deployment on embedded or mobile devices. In fact, it can run at >120FPS on Apple M1 CPU.
![macs](./image/calculate.png) 
- Experiments show that NanoTrack algorithm has good performance on tracking datasets.
    | Trackers            |   Backbone   | ModeSize | VOT2018 EAO | VOT2019 EAO | GOT-10k-Val AO | GOT-10k-Val SR | DTB70 Success | DTB70 Precision |
    | :-------------------------- | :----------: | :------: | :---------: | :---------: | :------------: | :------------: | :-----------: | :-------------: |
    | NanoTrack           | MobileNetV3  |  2.2MB   |    0.311    |    0.247    |     0.604      |     0.724      |     0.532     |      0.727      |
    | CVPR2021 LightTrack | MobileNetV3  |  7.7MB   |    0.418    |    0.328    |      0.75      |     0.877      |     0.591     |      0.766      |
    | WACV2022 SiamTPN    | ShuffleNetV2 |  62.2MB  |    0.191    |    0.209    |     0.728      |     0.865      |     0.572     |      0.728      |
    | ICRA2021 SiamAPN    |   AlexNet    | 118.7MB  |    0.248    |    0.235    |     0.622      |     0.708   |     0.585     |      0.786      |
    | IROS2021 SiamAPN++  |   AlexNet    |  187MB   |    0.268    |    0.234    |     0.635      |      0.73      |     0.594     |      0.791      |

- We provide [Android demo](https://github.com/HonglinChu/NanoTrack/tree/master/ncnn_android_nanotrack) and [MacOS demo](https://github.com/HonglinChu/NanoTrack/tree/master/ncnn_macos_nanotrack) based on ncnn inference framework. 

- We also provide [PyTorch code](https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack). It is friendly for training with much lower GPU memory cost than other models. NanoTrack only uses GOT-10k dataset to train, which only takes two hours on GPU3090.

# Mac

[PC demo](https://www.bilibili.com/video/BV1HY4y1q7B6?spm_id_from=333.999.0.0)


- 1. Modify your own CMakeList.txt

- 2. Build (Apple M1 CPU) 

    ```
    $ sh make_macos_arm64.sh 
    ```

# Android

[Android demo](https://www.bilibili.com/video/BV1eY4y1p7Cb?spm_id_from=333.999.0.0)

- 1. Modify your own CMakeList.txt

- 2. [Download](https://pan.baidu.com/s/1Yu1bpSKG-02fC5qekWXcLw)(password: 6cdd) OpenCV and NCNN libraries for Android 

# Reference  

https://github.com/Tencent/ncnn

https://github.com/Z-Xiong/LightTrack-ncnn

https://github.com/FeiGeChuanShu/ncnn_Android_LightTrack
