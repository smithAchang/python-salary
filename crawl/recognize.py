import os,sys,shutil
import cv2 as cv
import numpy as np

import unittest

if __name__ == '__main__':
    parentDirPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not parentDirPath in sys.path:
      sys.path.append(parentDirPath)


# 加载自身模块代码
from crawl.colorlog import ColorLog



# class定义
class CanNotRecognizeSelfError(Exception):
    pass

class DenyOtherDigitsError(Exception):
    pass


class RecognizeDigitCheckCode():
    def __init__(self):
        aiHome = os.path.join(os.getcwd(), 'ai')
        self.svmLhs  = cv.ml.SVM_load(os.path.join(aiHome,  'lhs.dat'))
        self.svmRhs  = cv.ml.SVM_load(os.path.join(aiHome,  'rhs.dat'))
        self.svmOper = cv.ml.SVM_load(os.path.join(aiHome,  'operator.dat'))

    def predict(self, img):
        img                     = processOrigiImg(img)
        lhsImg, rhsImg, operImg = splitOrigImg(img)

        lhsResult               = svmPredictResult(self.svmLhs,  lhsImg)
        rhsResult               = svmPredictResult(self.svmRhs,  rhsImg)
        operResult              = svmPredictResult(self.svmOper, operImg)

        if operResult == 1:
            operResult  = "+"
        else:
            operResult  = "-"

        ColorLog.info("lhsResult: %d , operResult: %s , rhsResult: %d ..."%(lhsResult, operResult, rhsResult))
        
        return lhsResult, rhsResult, operResult

    def predictSavedFile(self, pathtofile):
        img = cv.imread(pathtofile)

        lhsResult, rhsResult, operResult =  self.predict(img)

        return eval("%d%s%d"%(lhsResult, operResult, rhsResult))

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



# 门限百分比 0.82代表产生模板时使用比较弱的统计条件
__RECOGNIZE_THRESHOLD__ = {
                          0: (0.83, 0.97),
                          1: (0.0, 0.0),
                          2: (0.83, 0.97),
                          3: (0.83, 0.97),
                          4: (0.82, 0.95),
                          5: (0.83, 0.97),
                          6: (0.82, 0.95),
                          7: (0.83, 0.97),
                          8: (0.82, 0.95),
                          9: (0.82, 0.95),

}

__DENEY_THRESHOLD__ = {
                          0: (0.93, 0.82),
                          1: (0.0, 0.0),
                          2: (0.85, 0.82),
                          3: (0.83, 0.82),
                          4: (0.82, 0.81),
                          5: (0.83, 0.82),
                          6: (0.82, 0.81),
                          7: (0.83, 0.82),
                          8: (0.82, 0.81),
                          9: (0.82, 0.81),

}

# template 对于小幅的图片并不友好
# minusTemplate = cv.imread('./split/minusOper/oper2image.png')
# plusTemplate  = cv.imread('./split/plusOper/oper1image.png')
# zeroTemplate  = cv.imread('./split/0/lhs3image.png')
# oneTemplate   = cv.imread('./split/0/lhs3image.png')
# threeTemplate = cv.imread('./split/3/lhs17image.png')

# template      = minusTemplate

arithmeticGraphLeftSectionSplitPos       = 25
arithmeticGraphRightSectionSplitStartPos = 12
arithmeticGraphRightSectionSplitEndPos   = 38


methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

def doubleRecognizeOperTemplate(img, template):
   rightSectionPercent = recognizeTemplate(img[:, :arithmeticGraphLeftSectionSplitPos], template)
   leftSectionPercent  = recognizeTemplate(img[:, arithmeticGraphRightSectionSplitStartPos:arithmeticGraphRightSectionSplitEndPos], template)

   if leftSectionPercent == rightSectionPercent and leftSectionPercent > 0.9:
     return leftSectionPercent

   return 0.0

def recognizeLhsTemplate(img, template):
   testImage     = splitImages.specialProcessArithmeticCheckcodeImage(img)
   return recognizeTemplate(testImage[:, :arithmeticGraphLeftSectionSplitPos], template)

def recognizeRhsTemplate(img, template):
   testImage     = splitImages.specialProcessArithmeticCheckcodeImage(img)
   return recognizeTemplate(testImage[:, arithmeticGraphRightSectionSplitStartPos:arithmeticGraphRightSectionSplitEndPos], template)

def splitOrigImg(img):

    origi_lhs_split_end     = 12

    origi_rhs_split_begin   = 25
    origi_rhs_split_end     = 40

    origi_oper_split_begin  = 12
    origi_oper_split_end    = 26

    lhs  = img[:, :origi_lhs_split_end]
    rhs  = img[:,  origi_rhs_split_begin:origi_rhs_split_end]
    oper = img[:,  origi_oper_split_begin:origi_oper_split_end]

    return (lhs, rhs , oper)


def processOrigiImg(img):
       
    opers       = ImageOpers(img)
    img        = opers.gray().get()
    img        = opers.blur().get()
    img       = opers.threshold().get()
  
    return img

def splitAllOrigCapture():
    
    splitHome = os.path.join(os.getcwd(), 'split')
    shutil.rmtree(splitHome)
    
    os.makedirs(os.path.join(splitHome, 'lhs'),  exist_ok=True)
    os.makedirs(os.path.join(splitHome, 'rhs'),  exist_ok=True)
    os.makedirs(os.path.join(splitHome, 'oper'), exist_ok=True)

    def loadImgToProcess(home, filename):
        img = cv.imread(os.path.join(home, filename))
        img = processOrigiImg(img)
        lhs, rhs, oper = splitOrigImg(img)

        baseName = filename
        if baseName.find('.') != -1:
            baseName = baseName[:baseName.find('.')]

        cv.imwrite(os.path.join(splitHome, 'lhs',  "%s.jpg"%baseName), lhs)
        cv.imwrite(os.path.join(splitHome, 'rhs',  "%s.jpg"%baseName), rhs)
        cv.imwrite(os.path.join(splitHome, 'oper', "%s.jpg"%baseName), oper)



    walkFiles(os.path.join(os.getcwd(), 'authImg'), loadImgToProcess)

def svmAI(trainData, train_labels, savedFile='svm_data.dat'):
    svmKernel = cv.ml.SVM_LINEAR
    
    svm       = cv.ml.SVM_create()
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setType(cv.ml.SVM_C_SVC)
    
    # all svm needs set this para
    svm.setC(2.67)

    if svmKernel == cv.ml.SVM_RBF:
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


def knnAI(trainData, train_labels, savedFile='knn_data.npz', k=5):
    
    knn = cv.ml.KNearest_create()
    knn.train(trainData, cv.ml.ROW_SAMPLE, train_labels)
    ret,result,neighbours,dist = knn.findNearest(trainData, k=k)

    # stat
    mask = result==train_labels 
    correct = np.count_nonzero(mask)
    print("knn stat:", correct*100.0/result.size)

    # Save the data
    np.savez(savedFile, train=trainData, train_labels=train_labels)

    # Now load the data
    with np.load(savedFile) as data:
        print( data.files )
        trainData_load    = data['train']
        train_labels_load = data['train_labels']

    knn2 = cv.ml.KNearest_create()
    knn2.train(trainData_load, cv.ml.ROW_SAMPLE, train_labels_load)

    ret,result,neighbours,dist = knn.findNearest(trainData, k=k)

    # stat
    mask = result==train_labels 
    correct = np.count_nonzero(mask)
    print("knn stats when reload:", correct*100.0/result.size)

def collectImg(home, file, collections):
    img = cv.imread(os.path.join(home, file), 0)
    collections.append(img)


def digitTrain(section):
    
    home         = os.getcwd()
    digitHome    = os.path.join(home, 'data', section)
    trainData    = None
    train_labels = None


    for digit in range(0, 10):
        images = []

        walkFiles(os.path.join(digitHome, '%d'%digit), lambda h,f: collectImg(h, f, images))  

        labels    = np.repeat(digit, len(images))[:, np.newaxis]
        nd_imges  = np.asarray(images)

        if trainData is not None: 
            trainData     = np.concatenate((trainData, nd_imges))
            train_labels  = np.concatenate((train_labels, labels))
        else:
            trainData     = nd_imges
            train_labels  = labels

    if trainData is not None and trainData.size > 0 : 
        trainData       = np.float32(trainData).reshape(trainData.shape[0], trainData.shape[1]*trainData.shape[2])

        print('trainData:', trainData.shape, train_labels.shape)
        svmAI(trainData, train_labels, os.path.join(home, 'ai', '%s.dat'%section))


def lhsTrain():
   digitTrain('lhs')

def rhsTrain():
   digitTrain('rhs')

def operatorTrain():
    plusImges   = []
    minusImges  = []
    home        = os.getcwd()

    walkFiles(os.path.join(home, 'data', '+'), lambda h,f: collectImg(h, f, plusImges))
    positive_labels = np.repeat(1, len(plusImges))[:, np.newaxis]

    walkFiles(os.path.join(home, 'data', '-'), lambda h,f: collectImg(h, f, minusImges))
    negative_labels = np.repeat(0, len(minusImges))[:, np.newaxis]
    
    nd_plusImges    = np.asarray(plusImges)
    nd_minusImages  = np.asarray(minusImges)
    
    trainData       = np.concatenate((nd_plusImges, nd_minusImages))
    train_labels    = np.concatenate((positive_labels, negative_labels))


    trainData       = np.float32(trainData).reshape(trainData.shape[0], trainData.shape[1]*trainData.shape[2])

    print(nd_plusImges.shape, nd_minusImages.shape, trainData.shape, train_labels.shape)
    svmAI(trainData, train_labels, os.path.join(home, 'ai', 'operator.dat'))

def operatorPredict(img, savedFile):

    img          = processOrigiImg(img)
    lhs,rhs,oper = splitOrigImg(img)

    nd_oper      = np.asarray(oper)
    nd_oper      = np.float32(nd_oper).reshape(-1, nd_oper.shape[0]*nd_oper.shape[1])
    svm2         = cv.ml.SVM_load(savedFile)

    result       = svm2.predict(nd_oper)[1]
    result       = np.int16(result)

    return result[0][0]

    # stat

def svmPredictResult(svm, img):
    nd_oper      = np.asarray(img)
    nd_oper      = np.float32(nd_oper).reshape(-1, nd_oper.shape[0]*nd_oper.shape[1])
    result       = svm.predict(nd_oper)[1]
    result       = np.int16(result)
    return result[0][0]


def svmExternalTest(svm, img, relustOper):
    relustOper(svmPredictResult(svm, img))
        
   
def operatorResultProc(result, home, file, toDir):
    if result == 1:
        shutil.copyfile(os.path.join(home,file), os.path.join(toDir, "+", file))
    else:
        shutil.copyfile(os.path.join(home,file), os.path.join(toDir, "-", file))

def digitResultProc(result, home, file, toDir):
  
    shutil.copyfile(os.path.join(home,file), os.path.join(toDir, "%d"% result, file))
  
        

def pngPredict():
    home         = os.getcwd()
    pngHome      = os.path.join(home, 'test', 'png')
    resultHome   = os.path.join(home, 'result', 'png')
    resultLhsHome   = os.path.join(home, 'result', 'png','lhs')
    resultRhsHome   = os.path.join(home, 'result', 'png','rhs')

    svmOper      = cv.ml.SVM_load(os.path.join(home, 'ai', 'operator.dat'))
    svmLhs       = cv.ml.SVM_load(os.path.join(home, 'ai', 'lhs.dat'))
    svmRhs       = cv.ml.SVM_load(os.path.join(home, 'ai', 'rhs.dat'))

    plusDir      = os.path.join(resultHome, '+')
    minusDir     = os.path.join(resultHome, '-')

    shutil.rmtree(resultHome)   

    os.makedirs(plusDir, exist_ok=True)
    os.makedirs(minusDir, exist_ok=True)
    os.makedirs(resultLhsHome, exist_ok=True)
    os.makedirs(resultRhsHome, exist_ok=True)

    for digit in range(0,10):
       os.makedirs(os.path.join(resultLhsHome, '%d'%digit), exist_ok=True)
       os.makedirs(os.path.join(resultRhsHome, '%d'%digit), exist_ok=True)          
      
   
    for h,childDirs,files in os.walk(pngHome):
        for file in files:
            img          = cv.imread(os.path.join(h, file))
            img          = processOrigiImg(img)
            lhs,rhs,oper = splitOrigImg(img)

            svmExternalTest(svmOper, oper,  lambda r: operatorResultProc(r, h, file, resultHome))
            svmExternalTest(svmLhs,  lhs,   lambda r: digitResultProc(r, h, file, resultLhsHome))
            svmExternalTest(svmRhs,  rhs,   lambda r: digitResultProc(r, h, file, resultRhsHome))


def recognizeTemplate(img, template):
    h,w,l       = template.shape[::]
    # All the 6 methods for comparison in a list
    topLeftSet  = set()
    topLeftList = [];

    for meth in methods:
    
         method = eval(meth)
         # Apply template Matching
         res    = cv.matchTemplate(img, template, method)
         min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
         
         # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
         if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
         else:
            top_left = max_loc
         
         bottom_right = (top_left[0] + w, top_left[1] + h)

         topLeftSet.add(top_left)
         topLeftList.append(top_left)
         #print("method:%-20s: top_left[0]:%d, top_left[1]:%d; bottom_right[0]:%d, bottom_right[1]:%d"%(meth, top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

    max_percent        = 0.0
    max_percent_unique = None

    for unique in topLeftSet:
        stat = [ x for x in topLeftList if x == unique ]
        print(stat)
        max_percent_tmp   = (len(stat)/float(len(methods)))
        if max_percent_tmp > 0.5:
          max_percent_unique     = unique
          max_percent            = max_percent_tmp
          #print("max_percent_unique[0]:%d, max_percent_unique[1]:%d, max_percent:%f"%(max_percent_unique[0], max_percent_unique[1], max_percent))         
          break

   

    return max_percent
    
def recognizeTemplateInDir(template, dirPath, walk):
  percents = []
  for home, dirs, files in os.walk(dirPath):  
     for filename in files:
       fullpath = os.path.join(home, filename)  
       print("\r\n\r\nrecognize file:%s"%fullpath)
       img      = cv.imread(fullpath)
       percent  = walk(img, template)  

       if percent > 0.5 :
        print("max_percent:%f"%percent)
       else:
        print("no any exceed 50% !")

       percents.append(percent)
  return percents


def walkFiles(dirPath, oper):
    for home, childDirs, files in os.walk(dirPath):  
        for childDir in childDirs:
            walk(childDir, oper)
        for filename in files:
            oper(home, filename)

def moveToDst(walkhome, file, fromHome, toDir):
        shutil.move(os.path.join(fromHome, file), os.path.join(toDir, file))

def copyLabelOperToTrainData():
    home    = os.getcwd()
    dsthome = os.path.join(home, 'data')

    os.makedirs(os.path.join(dsthome, '+'),  exist_ok=True)
    os.makedirs(os.path.join(dsthome, '-'),  exist_ok=True)
    

    
    def moveToDst(walkhome,file, fromHome,toDir):
        shutil.move(os.path.join(fromHome, file), os.path.join(toDir, file))

    walkFiles(os.path.join(home, 'test', '+'), lambda h,f: moveToDst(h,f, os.path.join(home, 'split', 'oper'), os.path.join(dsthome, '+')))
    walkFiles(os.path.join(home, 'test', '-'), lambda h,f: moveToDst(h,f, os.path.join(home, 'split', 'oper'), os.path.join(dsthome, '-')))


def copyLabelDigitToTrainData(section):
    home         = os.getcwd()

    fromDigitHome  = os.path.join(home, 'test', section)
    toDigithome    = os.path.join(home, 'data', section)
    splitHome      = os.path.join(home, 'split', section)

    shutil.rmtree(toDigithome)

    for digit in range(0,10):
        toDstHome =  os.path.join(toDigithome, '%d'%digit)
        os.makedirs(toDstHome,  exist_ok=True)
        walkFiles(os.path.join(fromDigitHome, '%d'%digit), lambda h,f: moveToDst(h,f, splitHome, toDstHome))
    

def selecGoodTemplateDigit(digit, digitImage):
  
  topHome      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  testHome     = os.path.join(topHome, 'authImg')

  left         = '%dleft'%digit
  right        = '%dright'%digit

  percents     = []

  for dir in (left, right):
   dirPath = os.path.join(testHome, dir)

   for home, childDirs, files in os.walk(dirPath):  
     for filename in files:

       fullpath    = os.path.join(home, filename)  
       testImage   = cv.imread(fullpath)
      
      
       if dir == left:
         percent = recognizeLhsTemplate(testImage, digitImage)
       else:
         percent = recognizeRhsTemplate(testImage, digitImage)

       if percent > 0.5 :
        print("file %s max_percent:%f"%(fullpath, percent))
       else:
        print("file %s no any exceed 50%%, percent:%f"%(fullpath, percent))

       percents.append(percent)

  return percents

def recognizeTemplateDigit(digit, digitImage, threshold):
  
  topHome      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  testHome     = os.path.join(topHome, 'authImg')

  left         = '%dleft'%digit
  right        = '%dright'%digit


  for dir in (left, right):
   dirPath = os.path.join(testHome, dir)

   for home, childDirs, files in os.walk(dirPath):  
     for filename in files:

       fullpath    = os.path.join(home, filename)  
       testImage   = cv.imread(fullpath)
      
      
       if dir == left:
         percent = recognizeLhsTemplate(testImage, digitImage)
       else:
         percent = recognizeRhsTemplate(testImage, digitImage)

       if percent > threshold[0] :
        ColorLog.info("\r\n recognize %s for %d "%(fullpath, digit))
        continue
       else:
        raise CanNotRecognizeSelfError(fullpath)

'''
  Test Case
'''

class TemplatesStat():
  def __init__(self):
     self.reset()
     
     
  def reset(self):

     self.selfMean   = []
     self.selfVar    = []
     self.selfStd     = []

     self.othersMean = []
     self.othersStd  = []
     self.othersVar  = []
     self.differenceMean = []

  def add(self, pathtofile, selfMean, selfVar, selfStd, othersMean, othersVar, othersStd):

     self.selfMean.append((selfMean, pathtofile))
     self.selfVar.append((selfVar, pathtofile))
     self.selfStd.append((selfStd, pathtofile))

     self.othersMean.append((othersMean, pathtofile))
     self.othersVar.append((othersVar, pathtofile))
     self.othersStd.append((othersStd, pathtofile))

  def sort(self):
     # 识别自身，应该均值最大，方差、标准最小
     self.selfMean.sort(key=lambda x: x[0],  reverse=True)
     self.selfVar.sort(key=lambda x:  x[0])
     self.selfStd.sort(key=lambda x:  x[0])

     # 其它识别，应该去掉均值（代表概率）最小，去掉均值大方差小，去掉方差大的识别偏差大、标准比较大
     self.othersMean.sort(key=lambda x: x[0])
     self.othersVar.sort(key=lambda x:  x[0],  reverse=True)
     self.othersStd.sort(key=lambda x:  x[0],  reverse=True)

     # 差值最大的
     self.differenceMean = [ (item[0][0] - item[1][0], item[0][1]) for item in zip(self.selfMean, self.othersMean)]
     self.differenceMean.sort(key=lambda x: x[0],  reverse=True)

     return self

  def selectMostFit(self):
     topNum              = 5

    

     topMeanFitSet       = { item[1] for item in self.selfMean[:topNum] }
     topVarFitSet        = { item[1] for item in self.selfVar[:topNum] }
     topStdFitSet        = { item[1] for item in self.selfStd[:topNum] }

     ColorLog.notice("\r\nself stats ........................")

     ColorLog.info(topMeanFitSet)
     ColorLog.info(topVarFitSet)
     ColorLog.info(topStdFitSet)

     ColorLog.notice("\r\nother stats ........................")

     topOthersMeanFitSet = { item[1] for item in self.othersMean[:topNum] }
     topOthersVarFitSet  = { item[1] for item in self.othersVar[:topNum] }
     topOthersStdFitSet  = { item[1] for item in self.othersStd[:topNum] }

     ColorLog.info(topOthersMeanFitSet)
     ColorLog.info(topOthersVarFitSet)
     ColorLog.info(topOthersStdFitSet)

     topMostFit = topMeanFitSet.intersection(topVarFitSet, topStdFitSet)
     
     ColorLog.info(topMostFit)
     topMostFit = topMostFit.difference(topOthersMeanFitSet)

     ColorLog.info(topMostFit)
     topMostFit = topMostFit.difference(topOthersVarFitSet)

     ColorLog.info(topMostFit)
     topMostFit = topMostFit.difference(topOthersStdFitSet)

     ColorLog.info(topMostFit)

     if len(topMostFit) == 0:
        ColorLog.error("can not find moust good candinate, use weak conditions .....")
        topMostFit = { item[1] for item in self.differenceMean[:2] }


     
     ColorLog.notice(topMostFit)

     return topMostFit




class TestRecognize(unittest.TestCase):

    def setUp(self):
        ColorLog.info('setUp setup recognize test ...')
        

    def tearDown(self):
        ColorLog.info('tearDown close recognize test ...')
    

    def notest_Operator(self):
        
        print("\r\n\r\nrecognize plus dir: ")

        stats = recognizeTemplateInDir(plusTemplate, './authImg/plusOper/', doubleRecognizeOperTemplate)

        print('recognize plus oper in plusOper Dir, stats_len:%d, unique value:%d'%(len(stats), len(set(stats))))

        print("\r\n\r\nrecognize in minus dir: ")

        stats = recognizeTemplateInDir(plusTemplate, './authImg/minusOper/', doubleRecognizeOperTemplate)

        print('recognize plus oper in minusOper Dir, minus stats_len:%d, unique value:%d'%(len(stats), len(set(stats))))



        # 以识别加号作为主要依据
        print("\r\n\r\nrecognize special image ")
        specialFile = './authImg/minusOper/26image.png'
        img         = cv.imread(specialFile)
        res         = doubleRecognizeOperTemplate(img, plusTemplate) 

        print("file:%s last percent:%f"%(specialFile, res)) 


        specialFile = './authImg/minusOper/44image.png'
        img         = cv.imread(specialFile)
        res         = doubleRecognizeOperTemplate(img, plusTemplate) 

        print("file:%s last percent:%f"%(specialFile, res)) 


        stats = recognizeTemplateInDir(zeroTemplate, './authImg/0right/', recognizeRhsTemplate)

        print('recognize zero in zero as rhs dir, stats_len:%d, unique value:%d'%(len(stats), len(set(stats))))

        print('\r\n\r\n ##########################################################')

        stats = recognizeTemplateInDir(threeTemplate, './authImg/0right/', recognizeRhsTemplate)

        print('recognize three in zero as rhs dir, stats_len:%d, unique value:%d'%(len(stats), len(set(stats))))

    def htest_select_most_fit_template(self):
      ColorLog.info('run select most fit template digit ...')

      testDigits          = list(range(2, 10))
      testDigits.insert(0, 0)

      for templateDigit in testDigits:
      
          templatesStat       = TemplatesStat()

          candinatesTemplateSource = os.path.join(parentDirPath, 'split', '%d'%templateDigit)

          for candinatesHome, childDirs, candinatesFiles in os.walk(candinatesTemplateSource):  

            for candinate in candinatesFiles:
              
              candinatesTemplatePath = os.path.join(candinatesHome, candinate)
              ColorLog.notice('======================= new loop based template %s file =======================\r\n\r\n'%(candinatesTemplatePath, ))
              templateImage          = cv.imread(candinatesTemplatePath)
             
              otherRecognizeStats = []

              for recognizeDigit in testDigits:
                
                if recognizeDigit == templateDigit:
                    continue

                percents               = selecGoodTemplateDigit(recognizeDigit, templateImage) 
                otherRecognizeStats.extend(percents)
                mean                   = np.mean(percents)
                unique_percents        = set(percents)

                othersMean             = np.mean(otherRecognizeStats)
                othersVar              = np.var(otherRecognizeStats)
                othersStd              = np.std(otherRecognizeStats)


                if len(unique_percents) == 1:
                   ColorLog.warning('there exist good template file %s'%candinatesTemplatePath)

                ColorLog.notice('based template %s file, targetDigit: %d to recognize other %d , stats_len:%d, unique value:%d, min: %f , max: %f , mean: %f\r\n\r\n'%(candinatesTemplatePath, templateDigit, recognizeDigit, len(percents), len(unique_percents), min(unique_percents), max(unique_percents), othersMean))
               
             

              recognizeDigit         = templateDigit
              percents               = selecGoodTemplateDigit(recognizeDigit, templateImage) 
              unique_percents        = set(percents)
            
   

              selfMean               = np.mean(percents)
              selfVar                = np.var(percents)
              selfStd                = np.std(percents)

              templatesStat.add(candinatesTemplatePath, selfMean, selfVar, selfStd, othersMean, othersVar, othersStd)

              ColorLog.notice('based template %s file, targetDigit: %d to recognize self %d in test dir. mean: %f !\r\ntats_len:%d, unique value:%d, min: %f , max: %f , mean: %f ;\r\nselfMean:%f selfVar:%f selfStd:%f\r\nothersMean:%f othersVar:%f  othersStd:%f\r\n\r\n'%(candinatesTemplatePath, templateDigit, recognizeDigit, selfMean, len(percents), len(unique_percents), min(unique_percents), max(unique_percents), selfMean, selfMean, selfVar, selfStd, othersMean, othersVar, othersStd))
          
          topFit = templatesStat.sort().selectMostFit()

          templateDir = os.path.join(parentDirPath, 'template','%d'%templateDigit)
          shutil.rmtree(templateDir)
          os.makedirs(templateDir, exist_ok=True)

          for i, item in enumerate(topFit):
            shutil.copyfile(item, os.path.join(templateDir, "{}{}.png".format(templateDigit, i)))

    def test_template_fit_testset(self):
        global __RECOGNIZE_THRESHOLD__
        topHome      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        templateHome = os.path.join(topHome, 'template')
 
        testDigits          = list(range(2, 10))
        testDigits.insert(0, 0)

        topHome      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        testHome     = os.path.join(topHome, 'authImg')

  
        # 每个测试图片过模板图片，模板识别后平均值需要过阈值
        for everyTestSetDigit in testDigits: 
            left         = '%dleft'%everyTestSetDigit
            right        = '%dright'%everyTestSetDigit
            
            for testDir in (left, right):
                dirPath = os.path.join(testHome, testDir)

                for home, childDirs, files in os.walk(dirPath):  
                    for filename in files:

                        fullpath    = os.path.join(home, filename)  
                        testImage   = cv.imread(fullpath)

                        for everyTemplateDigit in testDigits:
                            testTemplateSource = os.path.join(templateHome, '%d'%everyTemplateDigit)
                        
                            percents = []

                        
                            for testTemplateDigitHome, testTemplateChildDirs, testTemplateDigitFiles in os.walk(testTemplateSource): 

                                for testTemplateDigitFile in testTemplateDigitFiles:
                                    candinatesTemplatePath = os.path.join(testTemplateDigitHome, testTemplateDigitFile)
                                
                                    templateImage          = cv.imread(candinatesTemplatePath)
                                    ColorLog.info('test set image %s  to check template image %s fitness file for %d =======================\r\n\r\n'%(fullpath, candinatesTemplatePath, everyTestSetDigit))
      
                                    if testDir == left:
                                        percent = recognizeLhsTemplate(testImage, templateImage)
                                    else:
                                        percent = recognizeRhsTemplate(testImage, templateImage)
                                
                                    percents.append(percent)
                        
                            
                            mean         = np.mean(percents)
                            max_percent  = max(percents)

                            if everyTemplateDigit == everyTestSetDigit:
                                ColorLog.notice('test image %s  finsh to check all template fitness for %d =======================\r\n\r\n'%(fullpath, everyTestSetDigit))
                                if mean > __RECOGNIZE_THRESHOLD__[everyTestSetDigit][0] and max_percent > __RECOGNIZE_THRESHOLD__[everyTestSetDigit][1]:
                                    ColorLog.info("recognize %d successfully  mean: %f, max_percent: %f ,threshold: %f~%f"%(everyTestSetDigit, mean, max_percent, __RECOGNIZE_THRESHOLD__[everyTestSetDigit][0], __RECOGNIZE_THRESHOLD__[everyTestSetDigit][1]))
                                    continue

                                elif mean == max_percent:
                                    ColorLog.info("allow special case, recognize %d successfully  mean: %f, max_percent: %f ,threshold: %f~%f"%(everyTestSetDigit, mean, max_percent, __RECOGNIZE_THRESHOLD__[everyTestSetDigit][0], __RECOGNIZE_THRESHOLD__[everyTestSetDigit][1]))
                                    continue

                                else:
                                    raise CanNotRecognizeSelfError("recognize %d in failure, mean: %f , max_percent: %f , threshold: %f~%f"%(everyTestSetDigit, mean, max_percent,  __RECOGNIZE_THRESHOLD__[everyTestSetDigit][0], __RECOGNIZE_THRESHOLD__[everyTestSetDigit][1])) 
                            else:
                                ColorLog.notice('test image %s  finsh to check all template unfitness for %d  =======================\r\n\r\n'%(fullpath, everyTemplateDigit))
                                if  mean > __DENEY_THRESHOLD__[everyTestSetDigit][0] and max_percent > __DENEY_THRESHOLD__[everyTestSetDigit][1]:
                                    raise  DenyOtherDigitsError("recognize %d in failure, does not deny %d ! mean: %f , max_percent: %f , threshold: %f~%f"%(everyTestSetDigit, everyTemplateDigit, mean, max_percent,  __DENEY_THRESHOLD__[everyTestSetDigit][0], __DENEY_THRESHOLD__[everyTestSetDigit][1])) 

                                elif mean == max_percent:
                                    ColorLog.error("recognize %d in failure, does not deny %d !  mean: %f , max_percent: %f , threshold: %f~%f"%(everyTestSetDigit, everyTemplateDigit, mean, max_percent,  __DENEY_THRESHOLD__[everyTestSetDigit][0], __DENEY_THRESHOLD__[everyTestSetDigit][1]))  

                                else:
                                    ColorLog.info("recognize %d successfully, deny %d !  mean: %f , max_percent: %f , threshold: %f~%f"%(everyTestSetDigit, everyTemplateDigit, mean, max_percent,  __DENEY_THRESHOLD__[everyTestSetDigit][0], __DENEY_THRESHOLD__[everyTestSetDigit][1])) 
                                    continue 
    def test_digitcheckcode(self):
        homeAuth = os.path.join(os.getcwd(), 'authImg')
        ai = RecognizeDigitCheckCode()

        for home,childDirs,files in os.walk(homeAuth):
            for file in files:
                result = ai.predictSavedFile(os.path.join(home, file))
                print('file:', file, 'result:', result,  " . and  wait to next ...")

   
             





# 识别标准，必须存在1，而且大于标准，例如，0.86






if __name__ == '__main__':
    
   
    parentDirPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not parentDirPath in sys.path:
      sys.path.append(parentDirPath)

   
    # 加载自身模块代码
    from crawl.colorlog import ColorLog
    import crawl.splitImages as splitImages

    ColorLog.notice("test case begin to run ...")

    
    unittest.main()

    











      
    

