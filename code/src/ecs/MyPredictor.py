#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2018/3/19 0:03
# @Author  : Dongqiang Chen
# @Email   : 980221271@.com
# @File    : MyPredictor.py
# @Software: PyCharm Community Edition
from function_tools import *
from math import *

# 最简单的预测，直接取平均
def MyPredictor(train, TIME_DELTA, flavor_dict):
    result = {}
    for each in flavor_dict:
        result[each] = 0
    for each in train:
        if result.has_key(each[0]):
            result[each[0]] += 1
    train_long=date_diff(getDate(train[-1][1]),getDate(train[0][1]))+1
    rate = 1.0 * (TIME_DELTA+1) / train_long
    for each in result:
        result[each] = round(result[each] * rate)
    return result

#二次指数平滑预测算法，alpha为平滑常数
def dabao(trainData_dict, x):#此函数是将newTrain重新按x个数据打包成一个数据
    dabaoData ={}
    for ID in trainData_dict.keys():
        dabaoData[ID] = []
        sum = len(trainData_dict[ID])
        point =0
        while(point<sum):
            temp = 0
            for i in range(x):
                temp += trainData_dict[ID][point]
                point += 1
                if point >= sum:
                    temp = temp/(i+1)*x
                    break
            dabaoData[ID].append(temp)
    return dabaoData
def double_exponential_smoothing(newTrainData,alpha,predict_day,x,flavor_dict):
    predict_num={}
    trainData_dict1 ={}
    for ID in flavor_dict.keys():#得到需要预测的ID数据
        trainData_dict1[ID] = newTrainData[ID]
    trainData_dict2 = dabao(trainData_dict1,x)
    trainData_dict = mean_fliter(trainData_dict2, 3)#最好3
    for ID in trainData_dict.keys():
        error = 0.0
        S1 = trainData_dict[ID][0]
        S2 = trainData_dict[ID][0]
        a = trainData_dict[ID][0]
        b = 0.0
        for elem in trainData_dict[ID]:
            error += abs(a + b - elem)
            S1 = alpha * elem + (1 - alpha) * S1
            S2 = alpha * S1 + (1 - alpha) * S2
            a = 2 * S1 - S2
            b = alpha /(1 - alpha)*(S1 - S2)
        Vnum = round(a + b * predict_day)
        if Vnum <= 0:
            Vnum = 0
        predict_num[ID] = Vnum
        #print "%s  future %d day has : %d, error is %f"%(ID, predict_day, Vnum, error)
    return predict_num
#两个alpha
def double2_exponential_smoothing(newTrainData,alpha,i,alpha2,predict_day,x,flavor_dict):
    predict_num={}
    trainData_dict1 ={}
    j=0
    for ID in flavor_dict.keys():#得到需要预测的ID数据
        trainData_dict1[ID] = newTrainData[ID]
    trainData_dict2 = dabao(trainData_dict1,x)
    trainData_dict = mean_fliter(trainData_dict2, 3)#最好3
    for ID in trainData_dict.keys():
        if j == i:
            error = 0.0
            S1 = trainData_dict[ID][0]
            S2 = trainData_dict[ID][0]
            a = trainData_dict[ID][0]
            b = 0.0
            for elem in trainData_dict[ID]:
                error += abs(a + b - elem)
                S1 = alpha2 * elem + (1 - alpha2) * S1
                S2 = alpha2 * S1 + (1 - alpha2) * S2
                a = 2 * S1 - S2
                b = alpha2 / (1 - alpha2) * (S1 - S2)
            Vnum = round(a + b * predict_day)
            if Vnum <= 0:
                Vnum = 0
            predict_num[ID] = Vnum
        #print "%s  future %d day has : %d, error is %f"%(ID, predict_day, Vnum, error)
        else:
            error = 0.0
            S1 = trainData_dict[ID][0]
            S2 = trainData_dict[ID][0]
            a = trainData_dict[ID][0]
            b = 0.0
            for elem in trainData_dict[ID]:
                error += abs(a + b - elem)
                S1 = alpha * elem + (1 - alpha) * S1
                S2 = alpha * S1 + (1 - alpha) * S2
                a = 2 * S1 - S2
                b = alpha / (1 - alpha) * (S1 - S2)
            Vnum = round(a + b * predict_day)
            if Vnum <= 0:
                Vnum = 0
            predict_num[ID] = Vnum
        j += 1
    return predict_num
#改进二次指数平滑预测算法
def impove2_double_exponential_smoothing(newTrainData,alpha,beta,predict_day,x,flavor_dict):
    predict_num={}
    trainData_dict1 ={}
    for ID in flavor_dict.keys():#得到需要预测的ID数据
        trainData_dict1[ID] = newTrainData[ID]
    trainData_dict2 = dabao(trainData_dict1,x)
    trainData_dict = mean_fliter(trainData_dict2, 3)#最好3
    for ID in trainData_dict.keys():
        #error = 0.0
        S1 = trainData_dict[ID][0]
        S2 = trainData_dict[ID][0]
        a = trainData_dict[ID][0]
        b = 0.0
        for elem in trainData_dict[ID]:
            #error += abs(a + b - elem)
            S1 = alpha * elem + (1 - alpha) * S1
            S2 = beta * S1 + (1 - beta) * S2
            a = ((alpha*(beta**2)-alpha)*S1 +(beta-alpha*beta)*S2)/\
                (beta - alpha*(beta**2))
            b = alpha*beta*(S1-S2)/(beta-alpha-alpha*beta+alpha*(beta**2))
        Vnum = round(a + b * predict_day)
        if Vnum <= 0:
            Vnum = 0
        predict_num[ID] = Vnum
        #print "%s  future %d day has : %d, error is %f"%(ID, predict_day, Vnum, error)
    return predict_num

#采用增量数据的改进二次指数平滑算法
def delt_double_exponential_smoothing(deltTrainData,alpha,predict_day,x,flavor_dict):
    predict_num={}
    trainData_dict1 ={}
    for ID in flavor_dict.keys():#得到需要预测的ID数据
        trainData_dict1[ID] = deltTrainData[ID]
    trainData_dict2 = dabao(trainData_dict1,x)
    trainData_dict = mean_fliter(trainData_dict2,3)#3
    for ID in trainData_dict.keys():
        error = 0.0
        S1 = trainData_dict[ID][0]
        S2 = trainData_dict[ID][0]
        a = trainData_dict[ID][0]
        b = 0.0
        for elem in trainData_dict[ID]:
            error += abs(a + b - elem)
            S1 = alpha * elem + (1 - alpha) * S1
            S2 = alpha * S1 + (1 - alpha) * S2
            a = 2 * S1 - S2
            b = alpha /(1 - alpha)*(S1 - S2)
        Vnum = round(a + b * predict_day)
        predict_num[ID] = Vnum
        print "%s  future %d day has : %d, error is %f"%(ID, predict_day, Vnum, error)
    return predict_num
#改进二次平滑预测算法,alpha可变
def getRange(start,stop,step):
    list =[]
    i = 1
    while(start <= stop):
        list.append(start)
        start = start +i*step
    return list
def impove1_double_exponential_smoothing(newTrainData,predict_day,x,flavor_dict):
    predict_num={}
    trainData_dict1 ={}
    for ID in flavor_dict.keys():#得到需要预测的ID数据
        trainData_dict1[ID] = newTrainData[ID]
    trainData_dict2 = dabao(trainData_dict1,x)
    trainData_dict = mean_fliter(trainData_dict2,3)#3
    for ID in trainData_dict.keys():
        error_final = 10000#初始化
        a_final = 0.0
        b_final = 0.0
        for alpha in getRange(0,1,0.001):
            error = 0.0
            S1 = trainData_dict[ID][0]
            S2 = trainData_dict[ID][0]
            a = trainData_dict[ID][0]
            b = 0.0
            for elem in trainData_dict[ID]:
                error += abs(a + b - elem)
                S1 = alpha * elem + (1 - alpha) * S1
                S2 = alpha * S1 + (1 - alpha) * S2
                a = 2 * S1 - S2
                b = alpha /(1 - alpha)*(S1 - S2)
            if error < error_final:
                a_final = a
                b_final = b
                error_final = error
        Vnum = round(a_final + b_final * predict_day)
        if Vnum <= 0:
            Vnum = 0
        predict_num[ID] = Vnum
        #print "flavor%d  future %d day has : %d, error is %f"%(ID, predict_day, Vnum, error_final)
    return predict_num
###############################################回归树算法#####################################
#对newTrainData进行处理的到决策树需要的训练数据，trainDataTree为字典
#x为对前x个数据作为对当前数据的输入
def getTreeData1(newTrainData,x):
    trainDataTree = {}#{key:[[list,label],[],...],...},list has x feature varies
    inData = {}#将来预测的输入
    for ID in newTrainData:
        trainDataTree[ID] = []
        templist = newTrainData[ID][:x]
        for elem in newTrainData[ID][x:]:
            trainDataTree[ID].append(templist+[elem])
            templist.pop(0)
            templist.append(elem)
        inData[ID] = templist
    return trainDataTree, inData
#not only feature is pre x data ,but also including delt x[t]-x[t-1] has x-1
def getTreeData2(newTrainData,x):
    inData = {}
    trainData = {}#{key:[[list],[],...],...},list has x+x-1 feature varies
    for ID in newTrainData:
        trainData[ID] = []
        templist = newTrainData[ID][:x]
        deltlist = []
        i=1
        j =0
        while(i < x):
            deltlist.append(templist[i] - templist[j])
            i += 1
            j += 1
        for elem in newTrainData[ID][x:]:
            trainData[ID].append(templist+deltlist+[elem])
            templist.pop(0)
            templist.append(elem)
            deltlist.pop(0)
            deltlist.append(elem-templist[-2])
        inData[ID] = templist + deltlist
    return trainData, inData
#得到数据集中的label
def getDataLabel(dataSet):
    datalabel = []
    for elem in dataSet:
        datalabel.append(elem[-1])
    return datalabel
#get featIndex feature list
def getDatafeat(dataSet, featIndex):
    datafeat = []
    for elem in dataSet:
        datafeat.append(elem[featIndex])
    return datafeat
#数据集按特定特征，特征值进行切分
def binSplitData(dataSet,feature, value):
    dataSet1 = []
    dataSet2 = []
    for elem in dataSet:
        if elem[feature] > value:
            dataSet1.append(elem)
        else:
            dataSet2.append(elem)
    return dataSet1, dataSet2
#创建叶节点类型
def regLeaf(dataSet):
    dataLabel = getDataLabel(dataSet)
    return sum(dataLabel)*1.0/len(dataLabel)
#创建误差类型
def regErr(dataSet):
    dataLabel = getDataLabel(dataSet)
    mean_square_error = 0
    mean = sum(dataLabel)*1.0/len(dataLabel)
    for elem in dataLabel:
        mean_square_error += (elem - mean)**2
    mean_square_error = sqrt(mean_square_error*1.0/len(dataLabel))
    return mean_square_error
#选择最佳特征和切分点
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr,ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    inf = 1000000# define infinit
    dataLabel = getDataLabel(dataSet)
    if len(set(dataLabel)) == 1:# if all values are equal, exit
        return None, leafType(dataSet)
    m = len(dataSet)# the number of datas
    n = len(dataSet[0])-1 # the number of features
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n):
        for splitVal in set(getDatafeat(dataSet, featIndex)):
            dataSet1, dataSet2 = binSplitData(dataSet,featIndex,splitVal)
            if len(dataSet1) < tolN or len(dataSet2) < tolN: continue
            newS = errType(dataSet1) + errType(dataSet2)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if S - bestS < tolS:
        return None, leafType(dataSet)
    dataSet1, dataSet2 = binSplitData(dataSet,bestIndex,bestValue)
    if len(dataSet1) < tolN or len(dataSet2) < tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue
# 创建树
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitData(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree
################################################回归树进行预测###########
#判断是否是树
def isTree(obj):
    return (type(obj).__name__=='dict')
#回归数值
def regTreeEval(model, inDat):
    return float(model)
#回归树预测
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData, modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData, modelEval)
        else:
            return modelEval(tree['right'],inData)
#################################################决策树预测函数##############
#对newTrainData进行预测
#delt_Time为将多少个数据打包
#numFeature为前多少个数据作为特征
def predict_regreeTree(newTrainData, flavor_dict, delt_Time, numFeature=7, tolS=1, tolN=20):
    predict_num={}
    trainData_dict1 ={}
    for ID in flavor_dict.keys():#得到需要预测的ID数据
        trainData_dict1[ID] = newTrainData[ID]
    trainData_dict2 = dabao(trainData_dict1,delt_Time)
    #trainData_dict = mean_fliter(trainData_dict2, 3)#最好3，噪声过滤,二次
    trainData_dict = mean_fliter(trainData_dict2, 3.5)  # 最好3.5，噪声过滤,回归树
    All_trainData, All_inData = getTreeData1(trainData_dict, numFeature)
    #All_trainData, All_inData = getTreeData2(trainData_dict, numFeature)
    for ID in All_trainData:
        trainData = All_trainData[ID]
        inData = All_inData[ID]
        tree = createTree(trainData, ops=(tolS, tolN))
        print "tree is ",tree
        Vnum = treeForeCast(tree, inData)
        predict_num[ID] = round(Vnum)
    return predict_num
#对get2输入进行更新
def update2_inData(inData,x,Vnum):
    temp1 = inData[:x]
    temp2 = inData[x:]
    temp1.pop(0)
    temp1.append(Vnum)
    temp2.pop(0)
    temp2.append(Vnum - temp1[-2])
    new_inData = temp1 +temp2
    return new_inData
def update1_inData(inData,x,Vnum):
    temp1 = inData[:x]
    temp1.pop(0)
    temp1.append(Vnum)
    new_inData = temp1
    return new_inData
#不打包决策树
def predict2_regreeTree(newTrainData, flavor_dict, delt_Time, numFeature=7, tolS=1, tolN=20):
    predict_num={}
    trainData_dict1 ={}
    for ID in flavor_dict.keys():#得到需要预测的ID数据
        trainData_dict1[ID] = newTrainData[ID]
    trainData_dict = mean_fliter(trainData_dict1, 3)#最好3，噪声过滤,二次
    #trainData_dict = mean_fliter(trainData_dict2, 3.5)  # 最好3.5，噪声过滤,回归树
    All_trainData, All_inData = getTreeData1(trainData_dict, numFeature)
    #All_trainData, All_inData = getTreeData2(trainData_dict, numFeature)
    for ID in All_trainData:
        predict_num[ID]=0
        trainData = All_trainData[ID]
        inData = All_inData[ID]
        tree = createTree(trainData, ops=(tolS, tolN))
        for day in range(delt_Time):
            Vnum = treeForeCast(tree, inData)
            inData = update1_inData(inData,numFeature,Vnum)
            predict_num[ID] += Vnum
        predict_num[ID] = round(predict_num[ID])
    return predict_num
#差分数
def difference(dataSet,interval=1):
    diff = []
    for i in range(interval, len(dataSet)):
        value = dataSet[i] - dataSet[i - interval]
        diff.append(value)
    return diff
def predict_diff_regreeTree(newTrainData, flavor_dict, delt_Time, numFeature=2, tolS=1, tolN=20):
    predict_num={}
    trainData_dict1 ={}
    for ID in flavor_dict.keys():#得到需要预测的ID数据
        trainData_dict1[ID] = difference(newTrainData[ID])#一阶差分
    trainData_dict = mean_fliter(trainData_dict1, 3)#最好3，噪声过滤,二次

    #trainData_dict = mean_fliter(trainData_dict2, 3.5)  # 最好3.5，噪声过滤,回归树
    All_trainData, All_inData = getTreeData1(trainData_dict, numFeature)
    #All_trainData, All_inData = getTreeData2(trainData_dict, numFeature)
    for ID in All_trainData:
        predict_num[ID]=0
        trainData = All_trainData[ID]
        inData = All_inData[ID]
        tree = createTree(trainData, ops=(tolS, tolN))
        for day in range(delt_Time):
            Vnum = treeForeCast(tree, inData)
            inData = update1_inData(inData, numFeature, Vnum)
            predict_num[ID] += Vnum
        predict_num[ID] = round(predict_num[ID]+delt_Time*newTrainData[ID][-1])
        if predict_num[ID] < 0:
            predict_num[ID] = 0
    return predict_num
###################################################提升树###############################
def caloutput(dataSet):
    output = 0.0
    for each in dataSet:
        output += each[-1]
    if len(dataSet) != 0:
        output /= len(dataSet)
    return output
#树桩损失
def stumpErr(dataSet):
    label = getDataLabel(dataSet)
    output = caloutput(dataSet)
    err = 0.0
    for i in range(len(dataSet)):
        err += (label[i] - output)**2
    return err
#创建树桩
def stump_chooseBestSplit(dataSet):
    inf = 1000000# define infinit
    numSteps = 10.0
    n = len(dataSet[0])-1 # the number of features
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n):
        rangemin = min(getDatafeat(dataSet,featIndex))
        rangemax = max(getDatafeat(dataSet,featIndex))
        stepSize = (rangemax - rangemin)/numSteps
        for j in range(-1, int(numSteps)+1):
            threshVal = (rangemin + float(j)*stepSize)
            dataSet1, dataSet2 = binSplitData(dataSet, featIndex, threshVal)
            err = stumpErr(dataSet1) + stumpErr(dataSet2)
            if err < bestS:
                bestS = err
                bestIndex = featIndex
                bestValue = threshVal
    return bestIndex, bestValue

def treeStump(dataSet):
    feat, val = stump_chooseBestSplit(dataSet)
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitData(dataSet, feat, val)
    retTree['left'] = caloutput(lSet)
    retTree['right'] = caloutput(rSet)
    return retTree
def stumpTreeForeCast(tree,indata):
    featIndex = tree['spInd']
    splitVal = tree['spVal']
    if indata[featIndex] > splitVal:
        predict = tree['left']
    else:
        predict = tree['right']
    return predict
#计算残差,返回dataSet的label被残差替换的列表数据
def calResidual_DataSet(dataSet, retT):
    residualSet = dataSet
    for num,each in enumerate(dataSet):
        real = dataSet[num][-1]
        predict = 0.0
        for i in range(len(retT)):
            predict += stumpTreeForeCast(retT[i], each[:-1])
        residual = real - predict
        residualSet[num][-1] = residual
    return residualSet
#创建提升树
#将每一次得到的回归树放到retT列表中
def boostTree(dataSet,numIter=20):
    retT = []
    T = treeStump(dataSet)
    print "T:  ###",T
    retT.append(T)
    for iter in range(numIter):
        residualSet = calResidual_DataSet(dataSet,retT)
        T = treeStump(residualSet)
        retT.append(T)
    return retT
#单个预测函数
def boostForeCast(retT,inData):
    predict_num = 0
    for i in range(len(retT)):
        predict_num += stumpTreeForeCast(retT[i], inData)
    return predict_num
#提升树整体预测
def predict_boostTree(newTrainData, flavor_dict,delt_Time, predict_day=1, numFeature=7, numIter=20):
    predict_num={}
    trainData_dict1 ={}
    for ID in flavor_dict.keys():#得到需要预测的ID数据
        trainData_dict1[ID] = newTrainData[ID]
    trainData_dict2 = dabao(trainData_dict1, delt_Time)
    #trainData_dict = mean_fliter(trainData_dict2, 3)#最好3，噪声过滤,二次
    trainData_dict = mean_fliter(trainData_dict2, 3.5)  # 最好3.5，噪声过滤,回归树
    All_trainData, All_inData = getTreeData1(trainData_dict, numFeature)
    #All_trainData, All_inData = getTreeData2(trainData_dict, numFeature)
    for ID in All_trainData:
        predict_num[ID] = 0
        trainData = All_trainData[ID]
        print "traindATA IS :  ", trainData
        inData = All_inData[ID]
        retT = boostTree(trainData, numIter=numIter)
        print "boostTree",retT
        for day in range(predict_day):
            Vnum = boostForeCast(retT, inData)
            inData = update1_inData(inData, numFeature, Vnum)
            predict_num[ID] += Vnum
        predict_num[ID] = round(predict_num[ID])
    return predict_num
#######################################################KNN#############################################
def mean(list):
    retdata = sum(list)*1.0/len(list)
    return retdata
def var(list):
    le = len(list)
    mea = mean(list)
    va = 0.0
    for i in range(le):
        va += (list[i] - mea)**2
    va = va/(le - 1)
    return va
def getDataknn_mean_var(newTrainData,x):
    trainDataTree = {}#{key:[[list,label],[],...],...},list has x feature varies
    inData = {}#将来预测的输入
    for ID in newTrainData:
        trainDataTree[ID] = []
        templist = newTrainData[ID][:x]
        junzhi = mean(templist)
        fangcha = var(templist)
        for elem in newTrainData[ID][x:]:
            trainDataTree[ID].append(templist+[junzhi,fangcha,elem])
            templist.pop(0)
            templist.append(elem)
            junzhi = mean(templist)
            fangcha = var(templist)
        inData[ID] = templist + [junzhi, fangcha]
    return trainDataTree, inData
#KNN数据
def getDataKNN(newTrainData,x):
    inData = {}
    trainData = {}#{key:[[list],[],...],...},list has x+x-1 feature varies
    for ID in newTrainData:
        trainData[ID] = []
        templist = newTrainData[ID][:x]
        deltlist = []
        i=1
        j =0
        while(i < x):
            deltlist.append(templist[i] - templist[j])
            i += 1
            j += 1
        for elem in newTrainData[ID][x:]:
            trainData[ID].append(templist+deltlist+[elem])
            templist.pop(0)
            templist.append(elem)
            deltlist.pop(0)
            deltlist.append(elem-templist[-2])
        inData[ID] = templist + deltlist
    return trainData, inData
#KNN数据归一化
def autoNorm(trainData, inData):
    dataNorm = trainData
    indataNorm = inData
    for ID in trainData:
        for featIndex in range(len(trainData[ID][0])-1):
            mindata = min(getDatafeat(trainData[ID], featIndex)+[inData[ID][featIndex]])
            maxdata = max(getDatafeat(trainData[ID], featIndex)+[inData[ID][featIndex]])
            if maxdata - mindata != 0:
                indataNorm[ID][featIndex] = (inData[ID][featIndex] - mindata)*1.0/(maxdata - mindata)
            else:
                indataNorm[ID][featIndex] = 0
            for elemindex in range(len(trainData[ID])):
                if maxdata-mindata != 0:
                    dataNorm[ID][elemindex][featIndex] = (trainData[ID][elemindex][featIndex]-mindata)*1.0/(maxdata-mindata)
                else:
                    dataNorm[ID][elemindex][featIndex] = 0
    return dataNorm, indataNorm
#knn计算距离
def distance1(trainData, inData):
    retData = []#[[distance,label],[],[]]
    for elem in trainData:
        temp = 0.0
        i=0
        for featindex in range(len(inData)):
            i+=1
            temp += (elem[featindex] - inData[featindex])**2
        temp = sqrt(temp)
        retData.append([temp, elem[i]])
    return retData
def distcos(trainData, inData):
    retData = []#[[distance,label],[],[]]
    for elem in trainData:
        temp = 0.0
        i=0
        xnorm =0.0
        ynorm =0.0
        for featindex in range(len(inData)):
            i+=1
            temp += elem[featindex]*inData[featindex]
            xnorm += inData[featindex]**2
            ynorm +=  elem[featindex]**2
        xnorm = sqrt(xnorm)
        ynorm = sqrt(ynorm)
        temp = temp*1.0/(xnorm*ynorm)
        retData.append([temp, elem[i]])
    return retData
#k个值处理函数
def knn_mean_getresult(distDat_Sort,k):
    temp = 0.0
    for i in range(k):
        temp += distDat_Sort[i][1]
    temp = round(temp*1.0/k)
    return temp
#KNN预测
def predict_KNN(newTrainData, flavor_dict,delt_Time,k=3,numFeature=7,knn_getresult=knn_mean_getresult,distance=distance1):
    predict_num={}
    trainData_dict1 ={}
    for ID in flavor_dict.keys():#得到需要预测的ID数据
        trainData_dict1[ID] = newTrainData[ID]
    trainData_dict = dabao(trainData_dict1, delt_Time)
    All_trainData, All_inData = getTreeData2(trainData_dict, numFeature)
    trainData, inData = autoNorm(All_trainData, All_inData)
    for ID in trainData:
        distanceDat = distance(trainData[ID],inData[ID])
        distDat_Sort = sorted(distanceDat,key=lambda x:x[0],reverse=True)
        predict_num[ID] = knn_getresult(distDat_Sort, k)
    return predict_num

def predict_KNN2(newTrainData, flavor_dict,delt_Time,k=3,numFeature=7,knn_getresult=knn_mean_getresult,distance=distance1):
    predict_num={}
    trainData_dict1 ={}
    for ID in flavor_dict.keys():#得到需要预测的ID数据
        trainData_dict1[ID] = newTrainData[ID]
    trainData_dict = dabao(trainData_dict1, delt_Time)
    trainData, inData = getTreeData1(trainData_dict, numFeature)
    #trainData, inData = getDataknn_mean_var(trainData_dict, numFeature)
    for ID in trainData.keys():
        distanceDat = distance(trainData[ID],inData[ID])
        distDat_Sort = sorted(distanceDat,key=lambda x:x[0],reverse=True)
        predict_num[ID] = knn_getresult(distDat_Sort, k)
    return predict_num