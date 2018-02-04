#coding=utf-8
from ImageServer import ImageServer
import numpy as np

imageDirs = ["../data/images/lfpw/trainset/", "../data/images/helen/trainset/", "../data/images/afw/"]
boundingBoxFiles = ["../data/py3boxesLFPWTrain.pkl", "../data/py3boxesHelenTrain.pkl", "../data/py3boxesAFW.pkl"]

datasetDir = "../data/"

meanShape = np.load("../data/meanFaceShape.npz")["meanShape"]

trainSet = ImageServer(initialization='rect')#相当于没有用bbx做训练，直接用的特征点截取框
trainSet.PrepareData(imageDirs, None, meanShape, 0, 2, True)#准备好图片名list，对应图片landmark的list，和对应图片的bbx的list，和meanshape。令我疑惑的是，startIdx=100，nImgs=100000，,300W数据集可没有那么多图片
trainSet.LoadImages()#读取图片，并对每张图调整好meanShape
trainSet.GeneratePerturbations(10, [0.2, 0.2, 20, 0.25])#位移0.2，旋转20度，放缩+-0.25
# import pdb; pdb.set_trace()
trainSet.NormalizeImages()#去均值，除以标准差
# trainSet.Save(datasetDir)#保存成字典形式，key为'imgs'，'initlandmarks'，'gtlandmarks'

validationSet = ImageServer(initialization='box')
validationSet.PrepareData(imageDirs, boundingBoxFiles, meanShape, 0, 100, False)
validationSet.LoadImages()
validationSet.CropResizeRotateAll()
import pdb; pdb.set_trace()
validationSet.imgs = validationSet.imgs.astype(np.float32)
validationSet.NormalizeImages(trainSet)
validationSet.Save(datasetDir)