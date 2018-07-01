# coding=utf-8
#定义日期数据类型
import random
class Date:
    def __init__(self, Year, Month, Day):
        self.Year = Year
        self.Month = Month
        self.Day = Day
    def __eq__(self, other):#日期等号重载运算符
        return(self.Year == other.Year and self.Month == other.Month and \
               self.Day == other.Day)
    def __ne__(self, other):#日期不等号重载运算符
        if self == other:
            return False
        else:
            return True

#实现闰年的判断函数
def isLeapYear(_year):
    if _year % 400 == 0:
        return True         #year%400的余数为0则为闰年
    else:
        if _year % 100 == 0:
            return False  # year %100 的余数不为0则不为闰年
        else:
            if _year % 4 == 0:
                return True  #year %4 的余数为0则为闰年
            else:
                return False #year % 4 的余数不为0则不为闰年

#实现日期相差多少天函数
def date_diff(bigDate, smallDate):#bigDate和smallDate为Date类型
    day_num = 0
    while bigDate != smallDate:
        day_num += 1
        date_add_1(smallDate)
    return day_num

#实现日期加1函数
def date_add_1(date):#date为Date类型数据
    date_temp = date
    date_temp.Day += 1
    if date_temp.Month == 1 or date_temp.Month == 3 or date_temp.Month == 5 or \
        date_temp.Month == 7or date_temp.Month == 8or \
        date_temp.Month == 10or date_temp.Month == 12:#判断31天的月份
        if date_temp.Day > 31:#如果天数超变为Day变为1，月份Month加1
            date_temp.Day = 1
            date_temp.Month += 1
            if date_temp.Month > 12:#如果月份超Month变为1,Year加1
                date_temp.Month = 1
                date_temp.Year += 1
    elif date_temp.Month == 2:#判断2月
        if isLeapYear(date_temp.Month):
            if date_temp.Day > 29:
                date_temp.Day = 1
                date_temp.Month += 1
        else:
            if date_temp.Day > 28:
                date_temp.Day = 1
                date_temp.Month += 1
    elif date_temp.Month == 4 or date_temp.Month == 6 or \
                    date_temp.Month == 9 or date_temp.Month == 11:#判读30天月份
        if date_temp.Day > 30:
            date_temp.Day = 1
            date_temp.Month += 1
    return date_temp #返回加1后的日期


def getDate(datetime_str):
    year = int(datetime_str[:4])
    month = int(datetime_str[5:7])
    day = int(datetime_str[8:10])
    return Date(year,month,day)

# 把结果转换成字符串
def result2str(pre_result, opt_result):
    flavor_sum = 0
    for each in pre_result:
        flavor_sum += int(pre_result[each])
    result = str(flavor_sum) + '\n'
    for each in pre_result:
        result = result + each + ' ' + str(int(pre_result[each])) + '\n'
    result = result + '\n' + str(len(opt_result))
    for i, each in enumerate(opt_result):
        result += '\n'
        result += str(i)
        for k in each:
            result = result + ' ' + k + ' ' + str(int(each[k]))
    print result
    return result

#增加训练数据的处理函数，得到每种规格按时间连续的顺序每天有多少台存储为字典，key为flavorID
#value为下标时间（跨度为天），数据为当天台数的列表
def get_NewTrainData(oldtraindata):
    MAX_DAY = 10000#用来初始化每个规格在每天的台数
    index = 0  # 记录当前数据的下标（跨度为天）
    firstDataFlag = 0#第一个数据对predate初始化
    newTrainData = {}
    for elem in oldtraindata:
        Date_now = getDate(elem[1])
        if firstDataFlag == 0:#第一个数据初始化
            Date_pre = Date_now
            newTrainData[elem[0]] = [0] * MAX_DAY
            newTrainData[elem[0]][index] = 1
            firstDataFlag = 1#进入正常循环
        else:
            index = index + date_diff(Date_now, Date_pre)
            if elem[0] in newTrainData.keys():
                newTrainData[elem[0]][index] += 1
            else:
                newTrainData[elem[0]] = [0] * MAX_DAY
                newTrainData[elem[0]][index] = 1
    newTrain ={}
    for key in newTrainData.keys():
        newTrain[key] = newTrainData[key][:(index+1)]
    print "There are %d data in the newtrainData"%(index+1)
    return newTrain
#对newTrainData进行处理用后一项减前一项delt作为数据
def get_delt_TrainData(newTrainData):
    deltTrainData = {}
    for ID in newTrainData.keys():
        count = len(newTrainData[ID])
        deltTrainData[ID] = [0]*count
        for index in range(1,count):
            delt = newTrainData[ID][index]- newTrainData[ID][index-1]
            if delt >= 0:
                deltTrainData[ID][index] = delt
            else:
                deltTrainData[ID][index] = 0
    return deltTrainData
#噪声去除
#思路：取得给个规格非零台天数的平均值，大于这个值的3到5倍当做噪声用平均值取代
def mean_fliter(newtrainData_dict,da):
    fliter_trainData ={}
    mean = {}
    no_zero_trainData = {}
    for ID in newtrainData_dict.keys():
        fliter_trainData[ID] =[]
        no_zero_trainData[ID] = []
        count = 0
        sum = 0
        for each in newtrainData_dict[ID]:
            if each > 0:
                count += 1
                sum += each
                no_zero_trainData[ID].append(each)
        mean[ID] = sum / float(count)
    for ID in newtrainData_dict.keys():
        zhonglist = sorted(newtrainData_dict[ID])
        zhong = zhonglist[int(len(zhonglist)/2)]
        no_zero_zhonglist = sorted(no_zero_trainData[ID])
        no_zero_zhong = no_zero_zhonglist[int(len(no_zero_zhonglist)/2)]
#############################################目前最佳去噪##########
        for each in newtrainData_dict[ID]:
            if each > da * mean[ID]:
                #each = mean[ID]*1.55#1.55最好
                each = no_zero_zhong*1.94# 1.55最好#1.94为二次指数去噪
                #each = no_zero_zhong * 2.3# 2.3为回归树去噪
                #each = mean[ID]  # 1.55最好#1.94为二次指数去噪
            fliter_trainData[ID].append(each)
###########################################探索中###################
#        for each in newtrainData_dict[ID]:
#            if each > da * mean[ID]:
                #each = mean[ID]*1.55#1.55最好
#                each = zhong*1.94# 1.55最好#1.94
#            fliter_trainData[ID].append(each)
    return fliter_trainData
def mean_tree_fliter(newtrainData_dict,da):
    fliter_trainData ={}
    mean = {}
    no_zero_trainData = {}
    for ID in newtrainData_dict.keys():
        fliter_trainData[ID] =[]
        no_zero_trainData[ID] = []
        count = 0
        sum = 0
        for each in newtrainData_dict[ID]:
            if each > 0:
                count += 1
                sum += each
                no_zero_trainData[ID].append(each)
        mean[ID] = sum / float(count)
    for ID in newtrainData_dict.keys():
        zhonglist = sorted(newtrainData_dict[ID])
        zhong = zhonglist[int(len(zhonglist)/2)]
        no_zero_zhonglist = sorted(no_zero_trainData[ID])
        no_zero_zhong = no_zero_zhonglist[int(len(no_zero_zhonglist)/2)]
#############################################目前最佳去噪##########
        for each in newtrainData_dict[ID]:
            if each > da * mean[ID]:
                #each = mean[ID]*1.55#1.55最好
                #each = no_zero_zhong*1.94# 1.55最好#1.94为二次指数去噪
                each = no_zero_zhong * 2.3# 2.3为回归树去噪
                #each = mean[ID]  # 1.55最好#1.94为二次指数去噪
            fliter_trainData[ID].append(each)
###########################################探索中###################
#        for each in newtrainData_dict[ID]:
#            if each > da * mean[ID]:
                #each = mean[ID]*1.55#1.55最好
#                each = zhong*1.94# 1.55最好#1.94
#            fliter_trainData[ID].append(each)
    return fliter_trainData
#融合
def merge(predict1_num,predict2_num,w=0.1):
    predict_num = {}
    for ID in predict1_num:
        predict_num[ID] = round(w*predict1_num[ID] + (1-w)*predict2_num[ID])
    return predict_num

def rescmp(x,y):
    if x[0]>y[0]:
        return 1
    elif x[0]<y[0]:
        return -1
    elif x[0]==y[0]:
        if x[1]>=y[1]:
            return 1
        else:
            return -1
def uniquePathsWithObstacles(obstacleGrid):
    """
    :type obstacleGrid: List[List[int]]
    :rtype: int
    """
    M, N = len(obstacleGrid), len(obstacleGrid[0])
    dp = [1] + [0] * (N-1)
    for i in range(M):
        for j in range(N):
            if obstacleGrid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j-1]
    return dp[N-1]
if __name__ == "__main__":
    time='2015-01-01 19:03:34'
    mDate = getDate(time)

    time2 = '2016-01-06 19:03:34'
    mDate2 = getDate(time2)
    print date_diff(mDate2,mDate)
    print bin(3)
    a='12345678'
    print filter(str.isalnum,str(a)).lower()
    print a[::-1]
