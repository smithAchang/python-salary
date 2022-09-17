#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
  
'''

import os,sys
import numpy as np
import cv2 as cv

if __name__ == '__main__':
    
    homePath = os.getcwd()
    if homePath not in sys.path:
        sys.path.append(homePath)

from crawl.customlog import ColorLog

import unittest

class ImageOpers():
    def __init__(self, img):
        self.img = img

    def zoom(self, factors=(7,7)):
        self.img =  cv.resize(self.img, None, fx=factors[0], fy=factors[1], interpolation = cv.INTER_CUBIC )
        return self

    def threshold(self,):
        ret, self.img =  cv.threshold(self.img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        return self
    
    # 对于噪点比较不错
    def blur(self, filter=(3, 3)):
        self.img  =  cv.medianBlur(self.img, filter[0])
        return self

    def edges(self, para=(60, 255)):
         self.img = cv.Canny(self.img, *para)
         return self

    def gray(self):
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        return self

    # 拉普拉斯锐化
    def laplacian(self):
        self.img = cv.Laplacian(self.img, cv.CV_64F)
        return self

    def get(self):
        return self.img

class Stats():
    pass
 


def eraseNoise(path, candyPara=(100, 200)):
    basename = os.path.basename(path)
    if basename.find('.') != -1:
        basename = basename[:basename.find('.')]

    img  = cv.imread(path)
    gray =  cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, gray_thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY)
    blur = cv.blur(gray_thresh,(3,3))
    ret, blur_thresh = cv.threshold(blur,127,255,cv.THRESH_BINARY)

    cv.imwrite(os.path.join(os.path.dirname(path),"%s_blurthresh.png"%basename), blur_thresh)
    edges = cv.Canny(blur_thresh, *candyPara)
    cv.imwrite(os.path.join(os.path.dirname(path),"%s_blurthresh_edges.png"%basename), edges)

    zoom = cv.resize(blur_thresh, None,fx=7, fy=7, interpolation = cv.INTER_CUBIC )
    cv.imwrite(os.path.join(os.path.dirname(path),"%s_blurthresh_zoomcubic.png"%basename), zoom)
    edges = cv.Canny(zoom, *candyPara)
    cv.imwrite(os.path.join(os.path.dirname(path),"%s_blurthresh_zoomcubic_edges.png"%basename), edges)
  
  
def processOriginalImage(img):
    
    opers       = ImageOpers(img)
    gray        = opers.gray().get()
    blur        = opers.blur().get()
    img         = opers.laplacian().get()
  
    return img

def writeimg(tohome, basename, format, tips, seces):

    #cv.imwrite(os.path.join(tohome, "lhs_%s_%s.%s"%(basename, tips, format)),  seces[0])
    #cv.imwrite(os.path.join(tohome, "rhs_%s_%s.%s"%(basename, tips, format)),  seces[1])
    cv.imwrite(os.path.join(tohome, "oper_%s_%s.%s"%(basename, tips, format)), seces[2])

    #cv.imwrite(os.path.join(tohome, "%s_lhs_%s.%s"%(basename, tips, format)),  seces[0])
    #cv.imwrite(os.path.join(tohome, "%s_rhs_%s.%s"%(basename, tips, format)),  seces[1])
    #cv.imwrite(os.path.join(tohome, "%s_oper_%s.%s"%(basename, tips, format)), seces[2])

def splitOrigImage(img):
    origi_lhs_split_end     = 17
    origi_rhs_split_begin   = 36
    origi_rhs_split_end     = 53
    origi_oper_split_begin  = 18
    origi_oper_split_end    = 35

    lhs  = img[:, :origi_lhs_split_end]
    rhs  = img[:, origi_rhs_split_begin:origi_rhs_split_end]
    oper = img[:, origi_oper_split_begin:origi_oper_split_end]

    return (lhs, rhs , oper)


def splitZoomImage(img):
    lhs  = img[10:130, :230]
    rhs  = img[10:130, 130:360]
    oper = img[10:130, 130:240]

    return (lhs, rhs, oper)

  

def writeImages(home, file, tohome='./split', format='png'):
    
    basename = file
    if basename.find('.') != -1:
        basename = basename[:basename.find('.')]

    img         = cv.imread(os.path.join(home,file))
    
    opers       = ImageOpers(img)
    
    gray        = opers.gray().get()
    #gray_thresh = opers.threshold().get()
    blur                    = opers.blur().get()
    thresh                  = opers.threshold().get()
    laplacian               = opers.laplacian().get()
    #blur_thresh             = opers.threshold().get()
    #blur_thresh_zoom        = opers.zoom().get()
    #blur_thresh_zoom_edges  = opers.edges().get()

    # 非清晰下弄边不太成功
    #grayBranch        = ImageOpers(gray)
    #grayBranch_edges  = grayBranch.edges().get()

    blurBranch         = ImageOpers(blur)
    blurBranch2        = ImageOpers(blur)
    blur_edeges        = blurBranch2.edges().get()
    blur_zoom          = blurBranch.zoom().get()
    blur_zoom_edges    = blurBranch.edges().get()

    laplacianBranch    = ImageOpers(blur)
    laplacian_zoom          = laplacianBranch.zoom().get()
    laplacian_zoom_edges          = laplacianBranch.edges().get()


    

    

    #writeimg(tohome, basename, format, 'gray', splitOrigImage(gray))
    #writeimg(tohome, basename, format, 'gray_edges', splitOrigImage(grayBranch_edges))
    #writeimg(tohome, basename, format, 'gray_thresh', splitOrigImage(gray_thresh))


    #writeimg(tohome, basename, format, 'blur', splitOrigImage(blur))

    #writeimg(tohome, basename, format, 'blur_edeges', splitOrigImage(blur_edeges))
    #writeimg(tohome, basename, format, 'blur_zoom', splitZoomImage(blur_zoom))
    #writeimg(tohome, basename, format, 'blur_zoom_edges', splitZoomImage(blur_zoom_edges))
    
    #writeimg(tohome, basename, format, 'laplacian', splitOrigImage(laplacian))
    writeimg(tohome, basename, format, 'thresh', splitOrigImage(thresh))
    #writeimg(tohome, basename, format, 'laplacian_zoom', splitZoomImage(laplacian_zoom))
    #writeimg(tohome, basename, format, 'laplacian_zoom_edges', splitZoomImage(laplacian_zoom_edges))

    #writeimg(tohome, basename, format, 'blur_thresh_zoom', splitZoomImage(blur_thresh_zoom))
    #writeimg(tohome, basename, format, 'blur_thresh_zoom_edges', splitZoomImage(blur_thresh_zoom_edges))

def walkFiles(dir, callback=print):
    for home, childdirs,files in os.walk(dir):
        
        for  childdir in childdirs:
            walkFiles(os.path.join(home, childdir), callback)
        
        for file in files:
            callback(home, file)


def produceSplitImages():
    walkFiles(os.path.join(os.getcwd(),'capture'), writeImages)


def svmAI(trainData, train_labels,savedFile='svm_data.dat'):
    svm = cv.ml.SVM_create()
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(trainData, cv.ml.ROW_SAMPLE, train_labels)
    svm.save(savedFile)
    result = svm.predict(trainData)[1]

    # stat
    mask = result==train_labels 
    correct = np.count_nonzero(mask)
    print("stat:", correct*100.0/result.size)

    svm2   = cv.ml.SVM_load(savedFile)

    result = svm2.predict(trainData)[1]

    # stat
    mask = result==train_labels 
    correct = np.count_nonzero(mask)
    print("stats when reload:", correct*100.0/result.size)

def collectImg(home, file, collections):
    img = cv.imread(os.path.join(home, file), 0)
    collections.append(img)

def operatorTrain():
    plusImges   = []
    minusImges = []
    

    walkFiles(os.path.join(os.getcwd(),'data', '+'), lambda h,f: collectImg(h, f, plusImges))
    positive_labels = np.repeat(1, len(plusImges))[:, np.newaxis]

    walkFiles(os.path.join(os.getcwd(),'data', '-'), lambda h,f: collectImg(h, f, minusImges))
    negative_labels = np.repeat(0, len(minusImges))[:, np.newaxis]
    nd_plusImges   = np.asarray(plusImges)
    nd_minusImages = np.asarray(minusImges)
    
    trainData    = np.concatenate((nd_plusImges, nd_minusImages))
    train_labels = np.concatenate((positive_labels, negative_labels))


    trainData    = np.float32(trainData).reshape(trainData.shape[0], trainData.shape[1]*trainData.shape[2])

    print(nd_plusImges.shape, nd_minusImages.shape, trainData.shape, train_labels.shape)
    svmAI(trainData, train_labels)

    return trainData,train_labels, nd_plusImges,nd_minusImages


  
def eraseNoise2(path, candypara=(100, 200)):
    basename = os.path.basename(path)
    if basename.find('.') != -1:
        basename = basename[:basename.find('.')]

    img  = cv.imread(path)
    gray =  cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, gray_thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY)

    blur = cv.blur(gray_thresh,(3,3))
    ret, blur_thresh1 = cv.threshold(blur,127,255,cv.THRESH_BINARY)

    blur2 = cv.blur(blur_thresh1,(2,2))
    ret, blur_thresh2 = cv.threshold(blur2,127,255,cv.THRESH_BINARY)


    cv.imwrite(os.path.join(os.path.dirname(path),"%s_blurthresh.png"%basename), blur_thresh1)
    cv.imwrite(os.path.join(os.path.dirname(path),"%s_blurthresh_2.png"%basename), blur_thresh2)

    for i in range(1,3):
        blur_thresh_tmp=eval("blur_thresh%d"%i)
        edges = cv.Canny(blur_thresh_tmp, *candypara)
        cv.imwrite(os.path.join(os.path.dirname(path),"%s_blurthresh_edges%d.png"%(basename,i)), edges)

        zoom = cv.resize(blur_thresh_tmp, None,fx=7, fy=7, interpolation = cv.INTER_CUBIC )
        cv.imwrite(os.path.join(os.path.dirname(path),"%s_blurthresh_zoomcubic%d.png"%(basename,i)), zoom)
        edges = cv.Canny(zoom, *candypara)
        cv.imwrite(os.path.join(os.path.dirname(path),"%s_blurthresh_zoomcubic_edges%d.png"%(basename,i)), edges)

__methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

def recognizeTemplate(img, template):
    #print(img.shape, template.shape)
    h,w,l = template.shape[::]
    # All the 6 methods for comparison in a list
    top_left_set  = set()
    top_left_list = []

    for meth in __methods:
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        top_left_set.add(top_left)
        top_left_list.append(top_left)

    percent = 0.0
    for unique in top_left_set:
        stat    = [x for x in top_left_list if x == unique]

        tmp_percent = len(stat)/float(len(top_left_list))

        if tmp_percent > 0.5 :
            percent = tmp_percent
            break

    return percent



def testSetCheck(templateHome, templateFile, testSetDir, recognizeOper, recognizePercents):
    templateImgPath = os.path.join(templateHome, templateFile)
    templateImg     = cv.imread(templateImgPath)

    basename = templateFile
    if basename.find('.') != -1:
        basename = basename[:basename.find('.')]

    def testFile(testSetHome, testSetFile):
        testFilePath = os.path.join(testSetHome, testSetFile)
        percent = recognizeOper(testFilePath, templateImg, templateImgPath)
        

        if templateFile in recognizePercents:
            recognizePercents[templateFile].percents.append((percent, testSetFile))
        else:
            recognizePercents[templateFile] = Stats()
            recognizePercents[templateFile].percents = [(percent, testSetFile)]

    walkFiles(testSetDir, testFile)


def case(templateDir, testSetDir, recognizeOper,  positive=True):

    recognizePercents = {}
    walkFiles(templateDir, lambda h,f: testSetCheck(h, f, testSetDir, recognizeOper, recognizePercents))
    
 
    stats = []
    means = []
    stds  = []
    maxs  = []
    mins  = []

    for key, value in recognizePercents.items():
        percents   = [ x[0] for x in value.percents ]
        value.mean = np.mean(percents)
        value.std  = np.std(percents)
        value.max  = max(percents)
        value.min  = min(percents)

        means.append((value.mean, key))
        stds.append((value.std, key))
        maxs.append((value.max, key))
        mins.append((value.min, key))
    
     # 降序
    means.sort(key=lambda k:k[0], reverse=positive)
    
    # 标准差小
    stds.sort(key=lambda k:k[0])

    maxs.sort(key=lambda k:k[0], reverse=positive)
    mins.sort(key=lambda k:k[0], reverse= positive)

    print('means:', means)
    print('stds:', stds)

    # 选择top5
    topMeanSet = set([ x for x in means[:5] ])
    topStdSet  = set([ x for x in stds[:5] ])
    
    print('topMeanSet:', topMeanSet)
    print('topStdSet:', topStdSet)

    intersection = topMeanSet.intersection(topStdSet)
    print("intersection:", intersection)

        

'''
  Test Case
'''

def recognizeOperator(testFilePath, templateImg, templateImgPath):
    img = cv.imread(testFilePath)
    img = processOriginalImage(img)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    lhs, rhs, oper = splitOrigImage(img)

    oper_percent   = recognizeTemplate(oper, templateImg)
    lhs_percent    = recognizeTemplate(lhs, templateImg)
    rhs_percent    = recognizeTemplate(rhs, templateImg)

    print('testFilePath:', testFilePath,'templateImgPath:',templateImgPath, '\r\noper_percent:',oper_percent, 'lhs_percent:', lhs_percent, 'rhs_percent:', rhs_percent )
    unique_percent = {lhs_percent, rhs_percent}
    max_percent    = max(unique_percent)
    min_percent    = min(unique_percent)

    if max_percent == 1.0 and min_percent > 0.83:
        return max_percent

    return 0.0

class TestRecognize(unittest.TestCase):

    def setUp(self):
        ColorLog.info('setUp case ...')
       

    def tearDown(self):
        ColorLog.info('tearDown case ...')
   


    def ttest_MinusOper(self):
      
      case('./template/minus', './test/minus', recognizeOperator)  
    

    def test_MinusNegative(self):

      case('./template/minus', './test/plus', recognizeOperator, positive=False)   
        




if __name__ == '__main__':
    
    homePath = os.getcwd()
    if homePath not in sys.path:
        sys.path.append(homePath)

 
    ColorLog.info("test case begin to run ...")
    unittest.main()

