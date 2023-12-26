from copy import copy
from datetime import datetime

import pandas as pd


class ST:
    def __int__(self):
        pass
    def AB(self,data,typ,begin = 2000):
        '''
        选择所有是 AB 的年份，已经 -2
        :param data: set_index('证券代码')后的 data
        :param begin: 开始打标签年份
        :return: pd.Series
        证券代码
        代码       打标签年份
        4         1997
        4         2020
        5         2001
        '''
        def change(X):
            # 2013 年披露的 ST，给 2011 年的 Financial indicator 打标签
            if type(X) == str:
                return int(X[0:4]) - 5
            if type(X) == datetime:
                return X.year - 5
            else:
                return X - 5

        ab = copy(data[data['变动类型'] == typ ])
        ab.loc[:, '变动公布日期'] = ab.loc[:, '变动公布日期'].map(change)

        back = ab.loc[:,'变动公布日期']
        return back[back >= begin]

    def special(self,data,begin = 2000):
        '''
        选择所有有特殊情况的年份 ， 已经提前 -2
        :param data:  ST 所有特殊处理的数据
        :param begin: 有特殊处理情况开始的年份
        :return: pd.series
        证券代码
        代码      有特殊情况的年份
        4         2008
        4         2009
        4         2010
        4         2021
        '''
        def change(X):
            if type(X) == str:
                return int(X[0:4]) - 5
            if type(X) == datetime:
                return X.year - 5
            else:
                return X - 5
        sp = copy(data)
        sp.loc[:, '变动公布日期'] = sp.loc[:, '变动公布日期'].map(change)
        back = sp.loc[:,'变动公布日期']

        return back[back >= begin]




