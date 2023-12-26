import pandas as pd
'''
A代表正常上市，
B代表ST,D代表*ST,C代表PT,S代表暂停上市，
T代表退市整理期，X代表终止上市
'''
def industry(x):
    return x[0]
pd.set_option('display.max_rows', None)  # exhibit all columns
pd.set_option('display.max_columns', None)  # exhibit all rows
#data = pd.read_excel('negative_abad_C_09-22.xls').set_index('代码')
data = pd.read_excel('positive09-22.xls').set_index('代码')
data2 = pd.read_excel('negative09-22.xls').set_index('代码')
data.loc[:,'总资产'] = data.loc[:,'总资产']/100000000
data.loc[:,'行业代码'] = data.loc[:,'行业代码'].apply(industry)
data2.loc[:,'总资产'] = data2.loc[:,'总资产']/100000000
data2.loc[:,'行业代码'] = data2.loc[:,'行业代码'].apply(industry)

print(data['行业代码'].value_counts())
print(data2['行业代码'].value_counts())
'''
store = data[(data['总资产']<=100) & (data['总资产']>=0)]
store2 = data2[(data2['总资产']<=100) & (data2['总资产']>=0)]
#print(data2['总资产'])
print("风险->风险的公司","原本有:",len(data),"筛选后:",len(store))
print("风险->正常的公司","原本有:",len(data2),"筛选后:",len(store2))
'''
'''
del store['总资产']
store.to_excel('pos_10-50.xls')
'''