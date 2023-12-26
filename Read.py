import numpy
import pandas as pd


class Read:
    '''
    将  SPT 数据与财务数据对齐
    '''
    def __int__(self):
        pass

    def get( self, data ,month = 12):
        '''
        选择 12 月的数据，并保留年份，转换为 TimeStamp
           :param data: 原始财务数据
           :param month: 选择的月份
           :return: 根据月份选择的数据，并转换成为 年份
            行业代码  统计截止日期      速动比率
            代码
            1       J66    2009       NaN
            1       J66    2010       NaN
            1       J66    2011       NaN
            1       J66    2012       NaN
        '''
        def change(X):
           if type(X) == pd.Timestamp:
               return X.year
           else:
               return -1
        data['统计截止日期'] = pd.to_datetime(data['统计截止日期'])
        p = data[data['统计截止日期'].dt.month == month]
        p.loc[:,'统计截止日期'] = p.loc[:,'统计截止日期'].map(change)
        return p


    def negtive(self,ab,data,typ,repeat,save_path):
        idx = pd.Series(ab.index).unique()
        for i in idx:
            t = ab.loc[i]
            if i in data.index:
                fi = data.loc[i]
                if type(t) != pd.Series:
                    t = [t]
                else:
                    t = pd.Series(t.tolist()).unique().tolist()  # unique 因为 同一年的都要删的，重复的没有意义

                if type(fi['统计截止日期']) != numpy.int64:  # 当前编号的 FI 有多条数据
                    idp = [i for i in fi['统计截止日期'].tolist() if i in t]  # sp 中没有记录的时间 list
                    for k in idp:
                        l = fi.loc[fi['统计截止日期'] == k]
                        repeat = repeat.append(l)
                else:
                    if fi['统计截止日期']  in t:
                        repeat = repeat.append(fi)
            else:
                print('FI中找不到:', i)
        back = repeat.iloc[1:, :]
        back['label'] = typ
        back.to_excel(save_path)

    def positive(self,sp,data,repeat,save_path):
        '''
        保证  repeat 第一行有数据
        :param sp: special treatment 的历史数据
        :param data: Financial indicator 数据表
        :param repeat: 保存删完的正常公司数据
        :return: None
        '''
        idx = pd.Series(sp.index).unique()
        # 特殊股票编号 去重
        for i in idx:
            t = sp.loc[i]#['变动公布日期']   # i 编号股票 特殊处理的历史数据
            if i in data.index:
                '''
                处理逻辑: 
                sp 中可能存在 data中不存在 的代码
                可能目前该公司已经退市了，故直接省略掉。
                fi : Financial indicator Dataset 中所有 i 编号的数据
                i    : 股票编号 i
                '''

                fi = data.loc[i]    # 编号为 i 的 Financial clustering
                #print(fi)
                #t = t.tolist()      # 编号为 i 的 历史 SP 时间
                '''
                逻辑:
                t 可能是 numpt.int64 也可能是 Seires
                将其转换成列表
                '''
                if type(t) != pd.Series:
                    t = [t]
                else:
                    t = pd.Series(t.tolist()).unique().tolist() # unique 因为 同一年的都要删的，重复的没有意义
                '''----------------------------------------------'''
                if type(fi['统计截止日期']) != numpy.int64: # 当前编号的 FI 有多条数据

                    idp =  [i for i in fi['统计截止日期'].tolist() if i not in t ] # sp 中没有记录的时间 list
                    for k in idp:
                        l = fi.loc[fi['统计截止日期'] == k]
                        repeat = repeat.append(l)
                        # 往 表中加 这条正常的数据
                else: # 当前编号 FI 有一条数据
                    if fi['统计截止日期'] not in t:
                       # 加入没有 Special Treatment 的数据 逻辑同 上if
                        repeat = repeat.append(fi)
            else:
                # FI中没有 sp的编号，可能公司退市
                print('FI中找不到:',i)
        back = repeat.iloc[1:,:]
        back['label'] = 0
        back.to_excel(save_path)
