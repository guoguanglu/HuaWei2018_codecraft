#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 21:03
# @Author  : Dongqiang Chen
# @Email   : 980221271@.com
# @File    : test_offline.py
# @Software: PyCharm Community Edition
import ecs
import predictor


# 此文件只用于线下调试，避免敲命令行

def main():
    # ecsDataPath = '../../data/TrainData.txt' #sys.argv[1]
    # inputFilePath = '../../data/input.txt' # sys.argv[2]
    # resultFilePath = '../../data/output.txt' # sys.argv[3]

    ecsDataPath = '../../data/TrainData_2015.1.1_2015.2.19.txt'  # sys.argv[1]
    ecsDataPath = '../../data/mynewdata4.txt'  # sys.argv[1]
    inputFilePath = '../../data/input_5flavors_cpu_7days.txt'  # sys.argv[2]
    resultFilePath = '../../data/output_result.txt'  # sys.argv[3]

    ecs_infor_array = ecs.read_lines(ecsDataPath)
    input_file_array = ecs.read_lines(inputFilePath)
    predic_result = predictor.predict_vm(ecs_infor_array, input_file_array)
    if len(predic_result) != 0:
        ecs.write_result(predic_result, resultFilePath)
    else:
        predic_result.append("NA")
        ecs.write_result(predic_result, resultFilePath)

if __name__ == "__main__":
    print 'main function begin.'
    main()
    print 'main function end.'
