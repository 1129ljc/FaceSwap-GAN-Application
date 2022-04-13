# FaceSwap-GAN-Application
人脸交换算法，一对一训练模型

## 执行步骤

Jupyter执行ipynb文件或者Python执行py文件

1.MTCNN_video_face_detection_alignment：MTCNN模型逐帧识别人脸

2.prep_binary_masks：人脸对齐，识别双眼和嘴巴，制作眼睛和嘴巴掩膜

3.FaceSwap_GAN_v2.2_train_test：训练一对一人脸交换模型

4.FaceSwap_GAN_v2.2_video_conversion：交换人脸

## Johnson2Trump

![](./test_result/1.gif)

## 模型地址

https://drive.google.com/drive/folders/1xHtLi_sZxsOq1eyjlIUzV40WIftyaH8h

## 算法来源

https://github.com/shaoanlu/faceswap-GAN