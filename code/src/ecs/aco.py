# coding=utf-8
import random
import collections
class Ant():
    #allnum总虚拟机数量
    def __init__(self, allnum, cpu_num, mem_size, dick_size,\
                 flavor_dict, memxuniji,info, ob, alpha, beta):
        self.memxuniji = memxuniji
        self.info = info
        self.cpu_num = cpu_num
        self.mem_size = mem_size
        self.dick_size = dick_size
        self.flavor_info = flavor_dict
        self.res = ob
        self.allnum = allnum
        self.path = [0] * allnum#记录蚂蚁的路径
        self.sum_res = 0#蚂蚁一次遍历的资源度
        self.vis = [0] * allnum#已选虚拟机标志
        self.cur_xuniji = random.randint(0, allnum-1)
        self.path[0] = self.cur_xuniji
        self.vis[self.cur_xuniji] = 1
        self.moved_cnt = 1
        self.alpha = alpha
        self.beta = beta
        #蚂蚁路径主机装虚拟机情况
        self.result = [collections.defaultdict(lambda: 0)]
        self.result[0][memxuniji[self.cur_xuniji]] += 1
        #存主机未利用的资源
        self.unusedRes = [[cpu_num - flavor_dict[memxuniji[self.cur_xuniji]][0],\
                           self.mem_size - flavor_dict[memxuniji[self.cur_xuniji]][1]]]
    #虚拟机分配函数
    def assign(self, ID):
        flag = 0  # 判断是否在历史申请的主机上找到了可以存放当前虚拟机的主机
        for i, each in enumerate(self.unusedRes):
            if self.flavor_info[ID][0] < each[0] and self.flavor_info[ID][1] < each[1]:  # 在前面已经找到可以存放当前虚拟机的主机
                self.unusedRes[i][0] = each[0] - self.flavor_info[ID][0]
                self.unusedRes[i][1] = each[1] - self.flavor_info[ID][1]
                self.result[i][ID] += 1
                flag = 1
                break
        if flag == 0:  # 没有找到满足条件的主机，则开辟新主机
            self.unusedRes.append([self.cpu_num - self.flavor_info[ID][0], \
                                   self.mem_size - self.flavor_info[ID][1]])
            self.result.append(collections.defaultdict(lambda: 0))
            self.result[-1][ID] += 1
    #启化函数,加入后的资源利用率
    def qifafunction(self,ID):
        flag = 0  # 判断是否在历史申请的主机上找到了可以存放当前虚拟机的主机
        #前面虚拟机是否可以放
        sum_unused = 0.0
        for each in self.unusedRes:
            if self.flavor_info[ID][0] < each[0] and self.flavor_info[ID][1] < each[1]:  # 在前面已经找到可以存放当前虚拟机的主机
                flag = 1
            sum_unused += each[self.res]
        if self.res == 0:
            if flag == 1:
                fenmu = len(self.result)*self.cpu_num*1.0
                fenzi = fenmu - sum_unused + self.flavor_info[ID][0]
                resRate = fenzi/fenmu
            else:
                fenmu = (len(self.result)+1)*self.cpu_num*1.0
                fenzi = len(self.result)*self.cpu_num*1.0 + self.flavor_info[ID][0]
                resRate = fenzi/fenmu
        else:
            if flag == 1:
                fenmu = len(self.result)*self.mem_size*1.0
                fenzi = fenmu - sum_unused + self.flavor_info[ID][1]
                resRate = fenzi/fenmu
            else:
                fenmu = (len(self.result)+1)*self.mem_size*1.0
                fenzi = len(self.result)*self.mem_size*1.0 + self.flavor_info[ID][1]
                resRate = fenzi/fenmu
        return resRate
    #选择下一个城市
    def chooseNextxuniji(self):
        nextSelectV = -1# 下一个要选择的虚拟机,初始化为-1
        #计算当前虚拟机和没有选择的虚拟机的信息素总和
        dbTotal = 0.0
        #保存各个虚拟机被选择的概率
        prob = [0.0]* self.allnum
        for i in range(self.allnum):
            if self.vis[i] == 0:
                prob[i] = self.info[self.cur_xuniji][i]**self.alpha*self.qifafunction(self.memxuniji[i])**self.beta
                dbTotal += prob[i]
            else:
                prob[i] = 0
        #进行轮盘赌选择
        if dbTotal > 0.0:
            dbtemp = random.random()
            for i in range(self.allnum):
                if self.vis[i] == 0:
                    dbtemp -= prob[i]*1.0/dbTotal
                    if dbtemp < 0.0:
                        nextSelectV = i
                        break
        #如果信息素过小，小到比double还小
        if nextSelectV == -1:
            for i in range(self.allnum):
                if self.vis[i] == 0:
                    nextSelectV = i
                    break
        return nextSelectV
    #蚂蚁移动
    def Move(self):
        nextSelectV = self.chooseNextxuniji()
        self.assign(self.memxuniji[nextSelectV])#进行分配
        self.path[self.moved_cnt] = nextSelectV#更新路径
        self.vis[nextSelectV] = 1#更新存放标志
        self.cur_xuniji = nextSelectV#更新当前虚拟机
        self.moved_cnt += 1
    #蚂蚁进行搜索一次
    def Search(self):
        while self.moved_cnt < self.allnum:
            self.Move()
class aco_optimizer():
    def __init__(self, pre_result, cpu_num, mem_size, dick_size,\
                 flavor_dict, ob, alpha, beta, antnum,Q,ROU,ITER):#Q信息素残留参数
        #将每一天预测虚拟机进行标号，存放
        memxuniji = []
        for each in pre_result:
            for i in range(int(pre_result[each])):
                memxuniji.append(each)
        self.memxuniji = memxuniji
        #预测虚拟机的总cpu和mem
        sum_useRes = [0, 0]
        for each in pre_result:
            for i in range(int(pre_result[each])):
                sum_useRes[0] += flavor_dict[each][0]
                sum_useRes[1] += flavor_dict[each][1]
        self.sumUseRes = sum_useRes
        self.len = len(memxuniji)
        self.cpu_num = cpu_num
        self.mem_size = mem_size
        self.H_Info = [cpu_num,mem_size]
        self.dick_size = dick_size
        self.flavor_info = flavor_dict
        self.res = ob
        self.alpha = alpha
        self.beta = beta
        self.info = [[1]*self.len]*self.len
        self.antnum = antnum
        self.Q = Q
        self.ROU = ROU
        self.ITER = ITER
        #创建antnum数量的蚂蚁列表
        antlist = []
        for i in range(self.antnum):
            antlist.append(Ant(self.len,self.cpu_num,self.mem_size,\
                               self.dick_size,self.flavor_info,\
                               self.memxuniji,self.info,self.res,\
                               self.alpha,self.beta))
        self.antlist = antlist
    def updateInfo(self):
        tempInfo = [[0]*self.len]*self.len
        #遍历每只蚂蚁
        for i in range(self.antnum):
            for j in range(1,self.len):
                m = self.antlist[i].path[j]
                n = self.antlist[i].path[j-1]
                tempInfo[n][m] += self.Q*self.sumUseRes[self.res]*1.0/\
                                  (len(self.antlist[i].result)*self.H_Info[self.res])
        #更新信息素
        for i in range(self.len):
            for j in range(self.len):
                self.info[i][j] = self.ROU*self.info[i][j] + tempInfo[i][j]
    def Search(self):
        ant_best = self.antlist[0]#初始化最佳为第一个蚂蚁
        for i in range(self.ITER):
            for j in range(self.antnum):
                self.antlist[j].Search()
            #保存最佳结果
            for j in range(self.antnum):
                if len(ant_best.result)> len(self.antlist[j].result):
                    ant_best = self.antlist[j]
            self.updateInfo()
        return ant_best.result