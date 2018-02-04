#coding=utf-8

##测试部分的代码

import tensorflow as tf
import ImageServer
from models import DAN

testSet = ImageServer.Load(datasetDir + "challengingSet.npz")


def getLabelsForDataset(imageServer):
    nSamples = imageServer.gtLandmarks.shape[0]
    nLandmarks = imageServer.gtLandmarks.shape[1]

    y = np.zeros((nSamples, nLandmarks, 2), dtype=np.float32)
    y = imageServer.gtLandmarks

    return y.reshape((nSamples, nLandmarks * 2))

nSamples = testSet.gtLandmarks.shape[0]
imageHeight = testSet.imgSize[0]
imageWidth = testSet.imgSize[1]
nChannels = testSet.imgs.shape[1]

Xtest = testSet.imgs

Ytest = getLabelsForDataset(testSet)

meanImg = testSet.meanImg
stdDevImg = testSet.stdDevImg
initLandmarks = testSet.initLandmarks[0].reshape((-1))

dan = DAN(initLandmarks)


with tf.session() as sess:
    Saver = tf.train.Saver()
    Writer = tf.summary.FileWriter("logs/", sess.graph)

    Saver.restore(sess,'./Model/Model')
    print('Pre-trained model has been loaded!')
       
    # Landmark68Test(MeanShape,ImageMean,ImageStd,sess)
    errs = []

    for iter in range(135):
        
        TestErr = sess.run(dan['S2_Cost'],{dan['InputImage']:Xtest,dan['GroundTruth']:Ytest,\
            dan['S1_isTrain']:False,dan['S2_isTrain']:False})
        errs.append(TestErr)
        print('The mean error for image %d is: %f'.format{iter, TestErr})

    errs = np.array(errs)

    print('The overall mean error is: %f'.format{np.mean(errs)})

