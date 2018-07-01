# coding=utf-8
import collections
from function_tools import *
from aco import*
from MyPredictor import double_exponential_smoothing
from MyPredictor import double2_exponential_smoothing
from MyOptimizer import random_First_Fit
from MyOptimizer import SA
from MyPredictor import delt_double_exponential_smoothing
from MyPredictor import impove1_double_exponential_smoothing
from MyPredictor import impove2_double_exponential_smoothing
from MyPredictor import MyPredictor
from MyPredictor import predict_regreeTree
from MyPredictor import predict_boostTree
from MyPredictor import predict_diff_regreeTree
from MyPredictor import predict2_regreeTree
from MyPredictor import predict_KNN
from MyPredictor import predict_KNN2
from MyOptimizer import MyOptimizer
from MyOptimizer import First_Fit

def predict_vm(ecs_lines, input_lines):
    result = []
    if ecs_lines is None:
        print 'ecs information is none'
        return result
    if input_lines is None:
        print 'input file information is none'
        return result

    #####################################训练数据处理start##################################################
    train = []
    for index, item in enumerate(ecs_lines):
        values = item.split("\t")
        flavorName = values[1]
        createTime = values[2]
        train.append([flavorName, createTime])
    newtrain_dict = get_NewTrainData(train)#newtrain_dict为历史数据中每个flavorID每天有多少台，是个字典
    deltTrainData_dict = get_delt_TrainData(newtrain_dict)#delt_TrainData_dict为新数据的增量数据
    #####################################输入数据处理start################################################
    input_data_len = len(input_lines)
    print input_data_len
    input_line_one = input_lines[0].split(' ')

    CPU = int(input_line_one[0])
    MEM = int(input_line_one[1]) * 1024
    HD = int(input_line_one[2].replace("\n", "")) * 1024
    FLAVOR_NUM = int(input_lines[2])
    print CPU, MEM, HD, FLAVOR_NUM

    FLAVOR_DATA = {}
    for index in range(3, input_data_len - 5):
        flavor_line = input_lines[index].replace("\n", "").split(" ")
        FLAVOR_DATA[flavor_line[0]] = [int(flavor_line[1]), int(flavor_line[2])]
    print FLAVOR_DATA

    res_type_str = input_lines[input_data_len - 4].replace("\n", "")
    RES_TYPE = 0 if res_type_str == 'CPU' else 1 # 0:CPU 1:MEM

    start_time_str = input_lines[input_data_len - 2].replace("\n", "")
    end_time_str = input_lines[input_data_len - 1].replace("\n", "")
    START_TIME = getDate(start_time_str)
    END_TIME = getDate(end_time_str)
    TIME_DELTA = date_diff(END_TIME,START_TIME)
    print RES_TYPE, START_TIME, END_TIME, TIME_DELTA

    #####################################预测与装箱部分##################################################

    # pre_result是预测的结果,opt_result是装箱的结果
    #pre_result = MyPredictor(train,TIME_DELTA, FLAVOR_DATA)
    pre_result =double_exponential_smoothing(newtrain_dict,0.322,1,TIME_DELTA,FLAVOR_DATA)#最佳0.321
    #pre_result = double2_exponential_smoothing(newtrain_dict, 0.322, 4, 0.322, 1, TIME_DELTA, FLAVOR_DATA)# all0.322最佳；分0.33,0.322
    #                                                                                                   0.322,0.325
    #pre_result2 = predict_regreeTree(newtrain_dict, FLAVOR_DATA, TIME_DELTA, numFeature=7,\
    #                               tolS=0.0001, tolN=4)
    #pre_result = merge(pre_result1,pre_result2, w=0.7)#合并二次和决策数w=0.7
    #pre_result = predict_KNN(newtrain_dict,FLAVOR_DATA,TIME_DELTA,k=2,numFeature=7)
    #pre_result = predict_KNN2(newtrain_dict, FLAVOR_DATA, TIME_DELTA, k=2, numFeature=7)
    #pre_result = predict_boostTree(newtrain_dict,FLAVOR_DATA,delt_Time=1,predict_day=TIME_DELTA,numFeature=7,numIter=6)
    #pre_result = predict2_regreeTree(newtrain_dict, FLAVOR_DATA, TIME_DELTA, numFeature=11,\
    #                               tolS=0.002, tolN=2)#不打包数据
    #pre_result = impove2_double_exponential_smoothing(newtrain_dict, 0.37,0.98,1, TIME_DELTA, FLAVOR_DATA)
    #pre_result = delt_double_exponential_smoothing(deltTrainData_dict, 0.4, 1, TIME_DELTA, FLAVOR_DATA)
    #pre_result = impove1_double_exponential_smoothing(newtrain_dict, 1, TIME_DELTA, FLAVOR_DATA)
    ##########################################东强分配#####################################
    opt = MyOptimizer(pre_result, CPU, MEM, HD, FLAVOR_DATA, RES_TYPE)
    opt.filling()
    opt_result = opt.get_result()
    ###########################################我的分配####################################
    #opt_result = First_Fit(pre_result,CPU,MEM,FLAVOR_DATA)
    #opt_result = random_First_Fit(pre_result, CPU, MEM, FLAVOR_DATA)
    #opt_result = SA(pre_result, CPU, MEM, FLAVOR_DATA)
    ##############################################蚁群分配#########################
    #opt2 = aco_optimizer(pre_result,CPU,MEM,HD,FLAVOR_DATA,RES_TYPE,alpha=1,\
    #                    beta=0.7,antnum=40,Q=1,ROU=0.9,ITER=1000)
    #opt_result = opt2.Search()
    result = result2str(pre_result, opt_result)

    return result
