# kaggle-rsna-pneumonia-detection-challenge

### CNN

- [關於影像辨識，所有你應該知道的深度學習模型 | Medium @2018-02-04](https://medium.com/@syshen/%E7%89%A9%E9%AB%94%E5%81%B5%E6%B8%AC-object-detection-740096ec4540)
- [一文讀懂：R-CNN、Fast R-CNN、Faster R-CNN、YOLO、SSD @2018-05-02](https://hk.saowen.com/a/ea0b8f4a0266432ae2df9b75548929b77393a26141d06a70f8a3061025462b77)
- [如何评价 Kaiming He 最新的 Mask R-CNN? | 知乎 @2017-03-23](https://www.zhihu.com/question/57403701)
- [Splash of Color: Instance Segmentation with Mask R-CNN and TensorFlow | Medium @2018-05-20](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)
- [Deep Learning in Computer Vision | Coursera](https://zh-tw.coursera.org/lecture/deep-learning-in-computer-vision/region-based-convolutional-neural-network-yU6QP)
- [mask rcnn解读](https://blog.csdn.net/u013010889/article/details/78588227)
- [14 Design Patterns To Improve Your Convolutional Neural Networks @2017-05-22](https://www.topbots.com/14-design-patterns-improve-convolutional-neural-network-cnn-architecture/)

### FCN
- [FCN 要點解釋](https://blog.csdn.net/Fate_fjh/article/details/53446630)
- [語義分割-全卷積神經網絡(FCN)](https://kknews.cc/zh-tw/news/kz9zalv.html)
- [FCN的学习及理解（Fully Convolutional Networks for Semantic Segmentation）](https://blog.csdn.net/qq_36269513/article/details/80420363)
- [FCN 简单梳理](https://blog.csdn.net/xg123321123/article/details/53092154)

### General NN
- [Weight initialization @2017-03-16](https://www.hksilicon.com/articles/1292771)
- [Advanced Tips for Deep Learning](https://www.youtube.com/watch?v=vVUkEbxltBw&feature=youtu.be): Batch Normalization | SELU | Hyperparameters
    

### Keras Tutorial

- [How to Use the Keras Functional API for Deep Learning @2017-10-27](https://machinelearningmastery.com/keras-functional-api-deep-learning/)
	
	```python
	from keras.models import Model
	from keras.layers import Input
	from keras.layers import Dense
	visible = Input(shape=(10,))
	hidden1 = Dense(10, activation='relu')(visible)
	hidden2 = Dense(20, activation='relu')(hidden1)
	hidden3 = Dense(10, activation='relu')(hidden2)
	output = Dense(1, activation='sigmoid')(hidden3)
	model = Model(inputs=visible, outputs=output)
	print(model.summary())
	```
	
	```
	Layer (type)                 Output Shape              Param #
	===========================================================================
	input_1 (InputLayer)         (None, 10)                0
	___________________________________________________________________________
	dense_1 (Dense)              (None, 10)                110 =(10+1)*10
	___________________________________________________________________________
	dense_2 (Dense)              (None, 20)                220 =(10+1)*20
	___________________________________________________________________________
	dense_3 (Dense)              (None, 10)                210 =(20+1)*10
	___________________________________________________________________________
	dense_4 (Dense)              (None, 1)                 11  =(10+1)*1
	===========================================================================
	Total params: 551
	Trainable params: 551
	Non-trainable params: 0
	```
	
- [How to calculate the number of parameters for convolutional neural network?](https://stackoverflow.com/a/42787467/9041712)

	```python
	from keras.models import Model
	from keras.layers import Input
	from keras.layers import Dense
	from keras.layers.convolutional import Conv2D
	from keras.layers.pooling import MaxPooling2D
	visible = Input(shape=(64, 64, 1))
	# fitler num: 32, filter shape: (4, 4)
	conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	hidden1 = Dense(10, activation='relu')(pool2)
	output = Dense(1, activation='sigmoid')(hidden1)
	model = Model(inputs=visible, outputs=output)
	print(model.summary())
	```
	
	```
	Layer (type)                 Output Shape              Param #
	===========================================================================
	input_1 (InputLayer)         (None, 64, 64, 1)         0
	___________________________________________________________________________
	# filter number 32, size 4*4. Say k=32, m=4, n=4, c=1
	conv2d_1 (Conv2D)            (None, 61, 61, 32)        544  = (4*4*1+1)*32
	                                    61=64-4+1                 (m*n*c+1)*k
	___________________________________________________________________________
	max_pooling2d_1 (MaxPooling2 (None, 30, 30, 32)        0
	                                    30=61/2
	___________________________________________________________________________
	# filter number 16, size 4*4. Say k=16, m=4, n=4, c=32
	conv2d_2 (Conv2D)            (None, 27, 27, 16)        8208 = (4*4*32+1)*16
	                                    27=30-4+1                 (m*n*c+1)*k
	___________________________________________________________________________
	max_pooling2d_2 (MaxPooling2 (None, 13, 13, 16)        0
	                                    13=27/2
	___________________________________________________________________________
	dense_1 (Dense)              (None, 13, 13, 10)        170  = (16+1)*10
	___________________________________________________________________________
	dense_2 (Dense)              (None, 13, 13, 1)         11
	===========================================================================
	Total params: 8,933
	Trainable params: 8,933
	Non-trainable params: 0
	```
- [keras版faster-rcnn算法详解](https://zhuanlan.zhihu.com/p/28585873)
- [Keras版Faster-RCNN代码学习](https://blog.csdn.net/qq_34564612/article/details/78881689)
- [Mask R-CNN tensorflow/keras的配置介紹、程式碼詳解與訓練自己的資料集演示](https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/515636/#outline__2_1_1)
	