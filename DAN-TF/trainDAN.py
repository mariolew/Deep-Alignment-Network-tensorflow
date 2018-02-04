#coding=utf-8

##训练部分的代码

import tensorflow as tf
import numpy as np

from ImageServer import ImageServer
from models import DAN

datasetDir = "../data/"
trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=40_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")
validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=9_perturbations=[]_size=[112, 112].npz")

# Create a new DAN regressor.
# 两个stage训练，不宜使用estimator
# regressor = tf.estimator.Estimator(model_fn=DAN,
# 	                                params={})

def getLabelsForDataset(imageServer):
    nSamples = imageServer.gtLandmarks.shape[0]
    nLandmarks = imageServer.gtLandmarks.shape[1]

    y = np.zeros((nSamples, nLandmarks, 2), dtype=np.float32)
    y = imageServer.gtLandmarks

    return y.reshape((nSamples, nLandmarks * 2))

nSamples = trainSet.gtLandmarks.shape[0]
imageHeight = trainSet.imgSize[0]
imageWidth = trainSet.imgSize[1]
nChannels = trainSet.imgs.shape[3]

Xtrain = trainSet.imgs
Xvalid = validationSet.imgs
# import pdb; pdb.set_trace()

Ytrain = getLabelsForDataset(trainSet)
Yvalid = getLabelsForDataset(validationSet)
testIdxsTrainSet = range(len(Xvalid))
testIdxsValidSet = range(len(Xvalid))
meanImg = trainSet.meanImg
stdDevImg = trainSet.stdDevImg
initLandmarks = trainSet.initLandmarks[0].reshape((1, 136))

dan = DAN(initLandmarks)

STAGE = 1

with tf.Session() as sess:
    Saver = tf.train.Saver()
    Writer = tf.summary.FileWriter("logs/", sess.graph)
    if STAGE < 2:
        sess.run(tf.global_variables_initializer())
    else:
        Saver.restore(sess,'./Model/Model')
        print('Pre-trained model has been loaded!')
       
    # Landmark68Test(MeanShape,ImageMean,ImageStd,sess)
    print("Starting training......")
    for epoch in range(1000):
        Count = 0
        while Count * 2 < Xtrain.shape[0]:
            RandomIdx = np.random.choice(Xtrain.shape[0],2,False)
            if STAGE == 1 or STAGE == 0:
                # sess.run(dan['S1_Optimizer'], feed_dict={dan['InputImage']:Xtrain[RandomIdx],\
                #     dan['GroundTruth']:Ytrain[RandomIdx],dan['S1_isTrain']:True,dan['S2_isTrain']:False})
                sess.run(dan['S1_Optimizer'], feed_dict={dan['InputImage']:Xvalid,\
                    dan['GroundTruth']:Yvalid,dan['S1_isTrain']:True,dan['S2_isTrain']:False})
            else:
                sess.run(dan['S2_Optimizer'], feed_dict={dan['InputImage']:Xtrain[RandomIdx],\
                    dan['GroundTruth']:Ytrain[RandomIdx],dan['S1_isTrain']:False,dan['S2_isTrain']:True})

            if Count % 8 == 0:
                TestErr = 0
                BatchErr = 0

                if STAGE == 1 or STAGE == 0:
                    TestErr = sess.run(dan['S1_Cost'],{dan['InputImage']:Xvalid,dan['GroundTruth']:Yvalid,\
                        dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                    BatchErr = sess.run(dan['S1_Cost'],{dan['InputImage']:Xtrain[RandomIdx],\
                        dan['GroundTruth']:Ytrain[RandomIdx],dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                else:
                    #Landmark,Img,HeatMap,FeatureUpScale =
                    #sess.run([Ret_dict['S2_InputLandmark'],Ret_dict['S2_InputImage'],Ret_dict['S2_InputHeatmap'],Ret_dict['S2_FeatureUpScale']],{Feed_dict['InputImage']:I[RandomIdx],Feed_dict['GroundTruth']:G[RandomIdx],Feed_dict['S1_isTrain']:False,Feed_dict['S2_isTrain']:False})
                    #for i in range(64):
                    #    TestImage = np.zeros([112,112,1])
                    #    for p in range(68):
                    #        cv2.circle(TestImage,(int(Landmark[i][p *
                    #        2]),int(Landmark[i][p * 2 + 1])),1,(255),-1)

                    #    cv2.imshow('Landmark',TestImage)
                    #    cv2.imshow('Image',Img[i])
                    #    cv2.imshow('HeatMap',HeatMap[i])
                    #    cv2.imshow('FeatureUpScale',FeatureUpScale[i])
                    #    cv2.waitKey(-1)
                    TestErr = sess.run(dan['S2_Cost'],{dan['InputImage']:Xvalid,dan['GroundTruth']:Yvalid,\
                        dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                    BatchErr = sess.run(dan['S2_Cost'],{dan['InputImage']:Xtrain[RandomIdx],\
                        dan['GroundTruth']:Ytrain[RandomIdx],dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                print('Epoch: ', epoch, ' Batch: ', Count, 'TestErr:', TestErr, ' BatchErr:', BatchErr)
            Count += 1
        Saver.save(sess,'./Model/Model')

