#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2018/3/19 0:10
# @Author  : Dongqiang Chen
# @Email   : 980221271@.com
# @File    : MyOptimizer.py
# @Software: PyCharm Community Edition
import collections
import random
from math import *
#定义[名称,cpu大小,内存大小,内存/cpu,需求,当前已装载个数,序号]
def rescmp(x, y, res):
    if res==0:
        if x[1]>y[1]:
            return 1
        elif x[1]<y[1]:
            return -1
        elif x[1]==y[1]:
            if x[2]>=y[2]:
                return 1
            else:
                return -1
    elif res==1:
        if x[2]>y[2]:
            return 1
        elif x[2]<y[2]:
            return -1
        elif x[2]==y[2]:
            if x[1]>=y[1]:
                return 1
            else:
                return -1


# 实现了最简单的二维装箱问题
class MyOptimizer():
    def __init__(self, pre_result, cpu_num, mem_size, dick_size, flavor_dict, ob):
        flavor_list = []
        for i, each in enumerate(flavor_dict):
            flavor_list.append(
                [each, flavor_dict[each][0], flavor_dict[each][1], flavor_dict[each][1] * 1.0 / flavor_dict[each][0],
                 pre_result[each], 0, i])
        # flavor_list是一个列表，记录了每种虚拟机的 [名称,cpu大小,内存大小,内存/cpu,需求,当前已装载个数,序号]，每次装箱都会变动该表
        self.flavor_list = flavor_list
        # 以下四个列表只是将flavor_list按不同方式排序，每次装箱不会变动表内信息
        self.flavor_list_sort_cpu = sorted(flavor_list, key=lambda x: x[1] * 1000000 + x[2])
        self.flavor_list_sort_mem = sorted(flavor_list, key=lambda x: x[2] * 1000000 + x[1])
        self.flavor_list_sort_rate_xiao = sorted(flavor_list, key=lambda x: x[3] * 1000000 - x[1])
        self.flavor_list_sort_rate_da = sorted(flavor_list, key=lambda x: x[3] * 1000000 + x[1], reverse=True)
        self.flavor_list_sort_minnum = sorted(flavor_list, key=lambda\
                x:min([cpu_num*1.0/x[1], mem_size*1.0/x[2]]))
        #按最大资源度排序最大放前面
        self.flavor_list_sort_maxRes = sorted(flavor_list, cmp= lambda x,y: rescmp(x,y,ob),reverse=True)

        self.cpu_num = cpu_num
        self.mem_size = mem_size
        self.dick_size = dick_size
        self.ob = ob
        self.wuli_rate = self.mem_size * 1.0 / self.cpu_num

        # 初始状态
        self.bin_num = 1
        self.bin_used_cpu = 0
        self.bin_used_mem = 0
        self.result = [collections.defaultdict(lambda: 0)]

    # 该函数决定每次将哪种虚拟机装入物理虚拟机内。是后面优化的核心函数。目前的策略是，尽可能保证已使用部分的cpu/mem的比率跟物理虚拟机的cpu/mem比率一致
    # ，然后优先装大虚拟机。
    def priority(self, ):
        # print self.flavor_list
        if self.bin_used_cpu != 0 and self.bin_used_mem * 1.0 / self.bin_used_cpu > self.wuli_rate:
            for each in self.flavor_list_sort_rate_xiao:
                if each[1] <= self.cpu_num - self.bin_used_cpu and each[2] <= self.mem_size - self.bin_used_mem and \
                                self.flavor_list[each[6]][4] > 0:
                    return each[6]
        else:
            for each in self.flavor_list_sort_rate_da:
                if each[1] <= self.cpu_num - self.bin_used_cpu and each[2] <= self.mem_size - self.bin_used_mem and \
                                self.flavor_list[each[6]][4] > 0:
                    return each[6]
        return -1
    def guo_priority(self,):
        if self.bin_used_cpu == 0:
            for each in self.flavor_list_sort_minnum:
                if each[1] <= self.cpu_num - self.bin_used_cpu and each[2] <= self.mem_size - self.bin_used_mem and \
                                self.flavor_list[each[6]][4] > 0:
                    return each[6]
        else:
            sort_closetowulirate = sorted(self.flavor_list,key=lambda x:\
                abs((self.bin_used_mem+x[2])*1.0/(self.bin_used_cpu+x[1])-\
                    self.wuli_rate))
            for each in sort_closetowulirate:
                if each[1] <= self.cpu_num - self.bin_used_cpu and each[2] <= self.mem_size - self.bin_used_mem and \
                                self.flavor_list[each[6]][4] > 0:
                    return each[6]
        return -1

    # 判断是否将虚拟机装完
    def is_ok(self):
        for each in self.flavor_list:
            if each[4] != 0:
                return 0
        return 1
    #按最大资源度排序装箱
    def maxRex_priority(self,):
        for each in self.flavor_list_sort_maxRes:
            if each[1] <= self.cpu_num - self.bin_used_cpu and each[2] <= self.mem_size - self.bin_used_mem and \
                            self.flavor_list[each[6]][4] > 0:
                return each[6]
        return -1
    def my_Filling(self):
        #记录所有主机中存放的虚拟机放置顺序
        memsort =[[]]
        #存放每个主机使用的（cpu，mem）
        memusedres = [[0, 0]]
        while self.is_ok() == 0:
            sel = self.priority()
            if sel == -1:
                for i in range(len(memsort)-2,-1,-1):#对前面的主机向后面主机放置
                    flag = 0
                    vlen = len(memsort[i])
                    for j in range(vlen - 1, -1, -1):#对主机中的虚拟机
                        for z in range(len(memsort)-1, i):#对后面的主机
                            if memsort[i][j][1] <= self.cpu_num - memusedres[z][0] and\
                                            memsort[i][j][2] <= self.mem_size - memusedres[z][1]:
                                memsort[z].append(memsort[i][j])#跟新memsort和memusedres
                                memusedres[z][0] += memsort[i][j][1]
                                memusedres[z][1] += memsort[i][j][2]
                                memusedres[i][0] -= memsort[i][j][1]
                                memusedres[i][1] -= memsort[i][j][2]
                                self.result[i][memsort[i][j][0]] -= 1
                                self.result[z][memsort[i][j][0]] += 1
                                memsort[i].pop(j)
                                flag = 1
                    if flag == 1:#再次试图填充
                        for each in self.flavor_list_sort_maxRes:
                            if each[1] <= self.cpu_num - memusedres[i][0] and\
                                            each[2] <= self.mem_size - memusedres[i][1] and \
                                            self.flavor_list[each[6]][4] > 0:
                                memsort[i].append(self.flavor_list[each[6]])#更新主机虚拟机
                                memusedres[i][0] += self.flavor_list[each[6]][1]#更新已经用的cpu和mem
                                memusedres[i][1] += self.flavor_list[each[6]][2]
                                self.flavor_list[each[6]][4] -= 1
                                self.flavor_list[each[6]][5] += 1
                                self.result[i][self.flavor_list[each[6]][0]] += 1
                if self.is_ok() == 0:#判断是否已经ok
                    self.bin_num += 1
                    self.result.append(collections.defaultdict(lambda: 0))
                    self.bin_used_cpu = 0
                    self.bin_used_mem = 0
                    memsort.append([])
                    memusedres.append([0, 0])
                    sel = self.maxRex_priority()
                else:
                    break
            self.bin_used_cpu += self.flavor_list[sel][1]
            self.bin_used_mem += self.flavor_list[sel][2]
            self.flavor_list[sel][4] -= 1
            self.flavor_list[sel][5] += 1
            self.result[self.bin_num - 1][self.flavor_list[sel][0]] += 1
            memsort[self.bin_num - 1].append(self.flavor_list[sel])
            memusedres[self.bin_num - 1][0] = self.bin_used_cpu
            memusedres[self.bin_num - 1][1] = self.bin_used_mem
    # 模拟了装载虚拟机的操作
    def filling(self):
        # 如果还有虚拟机要装
        while self.is_ok() == 0:
            # 选取一个能装到当前物理虚拟机的最优虚拟机
            sel = self.priority()
            #sel = self.guo_priority()
            # 如果没有可以选取的虚拟机，就申请一个新的物理虚拟机，继续选
            if sel == -1:
                self.bin_num += 1
                self.result.append(collections.defaultdict(lambda: 0))
                self.bin_used_cpu = 0
                self.bin_used_mem = 0
                #sel = self.priority()
                sel = self.guo_priority()
            # 将选取的虚拟机装入物理虚拟机
            self.bin_used_cpu += self.flavor_list[sel][1]
            self.bin_used_mem += self.flavor_list[sel][2]
            self.flavor_list[sel][4] -= 1
            self.flavor_list[sel][5] += 1
            self.result[self.bin_num - 1][self.flavor_list[sel][0]] += 1

    def get_result(self):
        return self.result

#First-Fit实现虚拟机部署，
#思路：把每一个虚拟机尽量放在前面已经防止的物理机里
#实在没有资源在开辟新的虚拟机
def assign(result, unusedResource, cpu_num, mem_size,ID, flavor_dict):
    i = 0   #指示当前主机
    flag = 0 # 判断是否在历史申请的主机上找到了可以存放当前虚拟机的主机
    new_unusedResource = unusedResource
    for each in unusedResource:
        if flavor_dict[ID][0] < each[0] and flavor_dict[ID][1] < each[1]:#在前面已经找到可以存放当前虚拟机的主机
            new_unusedResource[i][0] = each[0] - flavor_dict[ID][0]
            new_unusedResource[i][1] = each[1] - flavor_dict[ID][1]
            result[i][ID] += 1
            flag = 1
            break
        i += 1
    if flag == 0:#没有找到满足条件的主机，则开辟新主机
        unusedResource.append([cpu_num - flavor_dict[ID][0],\
                               mem_size - flavor_dict[ID][1]])
        result.append(collections.defaultdict(lambda: 0))
        result[-1][ID] += 1
    return result, unusedResource
def First_Fit(pre1_result,cpu_num,mem_size,flavor_dict):
    unusedResource = [[cpu_num, mem_size]]#用来存储每一个申请的主机的剩余资源
    result = [collections.defaultdict(lambda: 0)]
    pre_result = pre1_result.items()#对预测结果进行排序，按照主机单独存放数量最少的规格排序
    pre_result = sorted(pre_result,key=lambda x:min([cpu_num*1.0/flavor_dict[x[0]][0],\
                                                     mem_size*1.0/flavor_dict[x[0]][1]]))
    for elem in pre_result:
        pre_num = elem[1]
        while(pre_num):
            result, unusedResource = assign(result, unusedResource,cpu_num,\
                                            mem_size, elem[0], flavor_dict,)#分配完更新结果和未利用资源list
            pre_num -= 1
    return result
#随机排序first-fit
def random_First_Fit(pre_result,cpu_num,mem_size,flavor_dict):
    pre_result_ID_sort=[]
    for ID in pre_result:
        num = pre_result[ID]
        while(num):
            pre_result_ID_sort.append(ID)
            num -= 1
    VID_len = len(pre_result_ID_sort)
    iter = 1000
    pre_Hnum_final = 10000
    index_sort = range(VID_len)
    result_final =[]
    while(iter>0):
        random.shuffle(index_sort)#对下标随机排序
        unusedResource = [[cpu_num, mem_size]]  # 用来存储每一个申请的主机的剩余资源
        result = [collections.defaultdict(lambda: 0)]
        for index in index_sort:
            result, unusedResource = assign(result, unusedResource, cpu_num, \
                                            mem_size, pre_result_ID_sort[index]\
                                            , flavor_dict)
        pre_Hnum = len(result)
        if pre_Hnum < pre_Hnum_final:
            pre_Hnum_final = pre_Hnum
            result_final = result
        iter -= 1
    return result_final

def get_newsort(oldsort,VID_len):
    #random.shuffle(oldsort)#为重新sort
    #return oldsort
###############################################################
    #采用两交互与三交互进行跟新新的index_sort
    while(1):
        i = random.randint(0, VID_len-1)
        j = random.randint(0, VID_len-1)
        if i != j:
            break
    oldsort[i], oldsort[j] = oldsort[j], oldsort[i]
    '''    else:
        while(1):
            loc1 = random.randint(0, VID_len)
            loc2 = random.randint(0, VID_len)
            loc3 = random.randint(0, VID_len)
            if loc1 != loc2 and loc2!=loc3 and loc1!=loc3:
                break
        #满足 loc1<loc2<loc3
        if loc1 > loc2:
            loc1, loc2 = loc2, loc1
        if loc2 > loc3:
            loc2, loc3 = loc3, loc2
        if loc1 > loc2:
            loc1, loc2 = loc2, loc1
        #将[loc1, loc2)区间的数据插入到loc3之后
        tmplist = oldsort[loc1:loc2]'''
    return oldsort
#模拟退火算法
def SA(pre_result,cpu_num,mem_size,flavor_dict):
    T= 3000 #初始温度
    DELTA = 0.98 #温度衰减速度
    ILOOP = 1000#内循环次数
    EPS = 1e-7 #停止温度
    pre_result_ID_sort=[]
    for ID in pre_result:
        num = pre_result[ID]
        while(num):
            pre_result_ID_sort.append(ID)
            num -= 1
    VID_len = len(pre_result_ID_sort)
    init_index = range(VID_len)
    sort_newindex = init_index
    old_Hnum = 10000
    pre_Hnum_final = 10000
    result_final =[]
    while(1):
        for i in range(ILOOP):
            sort_newindex = get_newsort(sort_newindex,VID_len)
            unusedResource = [[cpu_num, mem_size]]  # 用来存储每一个申请的主机的剩余资源
            result = [collections.defaultdict(lambda: 0)]
            for index in sort_newindex:
                result, unusedResource = assign(result, unusedResource, cpu_num, \
                                                mem_size, pre_result_ID_sort[index] \
                                                , flavor_dict)
            new_Hnum = len(result)
            dE = new_Hnum - old_Hnum
            if dE < 0 or exp(-dE/T)>random.random():
                old_Hnum = new_Hnum
                old_result = result
        if old_Hnum < pre_Hnum_final:
            pre_Hnum_final = old_Hnum
            result_final = old_result
        if T < EPS:
            break
        T *= DELTA
    return result_final


