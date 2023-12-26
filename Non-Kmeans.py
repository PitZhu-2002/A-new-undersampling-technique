import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler,TomekLinks,EditedNearestNeighbours,RepeatedEditedNearestNeighbours,\
    AllKNN,CondensedNearestNeighbour,ClusterCentroids
from sklearn import tree
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE,RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from smote_variants import SMOTE_TomekLinks, SMOTE_ENN, Borderline_SMOTE1, Borderline_SMOTE2, ADASYN, AHC, LLE_SMOTE, \
    distance_SMOTE, SMMO, Stefanowski, Safe_Level_SMOTE,SVM_balance,AMSCO,SYMPROD,KernelADASYN,SOMO
from com.hdu.数据分析实战.Business_Analysis.Experiment.mode.Function import Function
from com.hdu.数据分析实战.Business_Analysis.Experiment.mode.支持向量机.Base_ratio import svm_undersampling
from com.hdu.数据分析实战.Business_Analysis.Experiment.mode.支持向量机.svm_undersampling_kmeans import \
    svm_undersampling_kmeans
import warnings

from com.hdu.数据分析实战.Business_Analysis.Experiment.mode.支持向量机.svm_undersampling_nonKmeans import \
    svm_undersampling_non_kmeans

warnings.filterwarnings("ignore")

def operate(data,random_state = 10,kmeans = False,cluster = 19):
    # 基分类器初始化
    base = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=10,min_samples_leaf=2,random_state=10)
    #base = KNeighborsClassifier(n_neighbors=3)
    # bootstrap == False 控制数据一样的变量
    if kmeans == False:
        svm_sam = svm_undersampling_non_kmeans(data = data,bootstrap = False,random_state=random_state)
    else:
        svm_sam = svm_undersampling_kmeans(data = data,bootstrap = False,cluster = cluster)
    svm_sam.generate()
    #print(svm_sam.prime.iloc[:,-1].value_counts())
    base.fit(svm_sam.prime.iloc[:,:-1] , svm_sam.prime.iloc[:,-1])
    return base

def start(name,data,save_path,rd_list,times=20,cluster=19,kmeans = True):
    PM_total = []                   # 对比方法 Auc、F-measure、G-measure、Recall、Specificity 集合
    ACC_total = []                  # 对比方法准确率
    #svm_PM = np.array([0]*5)        # Support Vector 欠采样 5个指标
    svmk_PM = [[] for i in range(6)]       # Support Vector K-means 5个指标
    #svm_ACC = []                    # Support Vector 欠采样准确率
    svmk_ACC = []                   # Support Vector K-means 准确率

    for i in range(len(name)):
        PM_total.append([[] for j in range(6)])
        #PM_total[i].append()  # 每个下标代表一个 PM
        ACC_total.append([])                # 每个下标代表一个 准确率

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    stf_split = StratifiedShuffleSplit(n_splits=times,test_size=0.1,random_state = 10)
    id = 0
    rd_i = 0
    for train_index, test_index in stf_split.split(X, y):
        X_prime, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_prime, y_test = np.array(y)[train_index], np.array(y)[test_index]
        #print(id)
        #print(pd.DataFrame(y_test).value_counts())
        id = id + 1
        pool_sampling = []
        pool_classifier = []  # 下标 表示分类器的种类，对应 name 里面过采样方法产生的数据
        pool_prd = []
        pool_ppb = []

        # base_svm = operate(data.iloc[train_index],kmeans=False)
        base_svmk = operate(data.iloc[train_index],kmeans=kmeans,cluster = cluster,random_state=rd_list[rd_i])
        rd_i = rd_i + 1
        back1 = Function.cal_F1_AUC_Gmean(y_test=y_test, y_pre=base_svmk.predict(X_test),
                                                    prob=Function.proba_predict_minority(base_svmk, X_test))
        for l in range(len(svmk_PM)):
            svmk_PM[l].append(back1[l])
        svmk_ACC.append(base_svmk.score(X_test, y_test))
    df = pd.DataFrame(columns=['Accuracy', 'F1','F2', 'Auc', 'G-mean', 'Recall', 'Precision'])
    df_roc = ['F1','F2','Auc', 'G-mean', 'Recall', 'Precision']
    #os.mkdir(str(cluster))
    if os.path.exists('Non-Kmeans/Non-Kmeans_LINEAR10'):
        print('存在')
    else:
        os.mkdir('Non-Kmeans/Non-Kmeans_LINEAR10')
    per2 = pd.DataFrame(columns=[i for i in range(times)])
    per2.loc['Acc'] = svmk_ACC
    for p in range(len(svmk_PM)):
        per2.loc[df_roc[p]] = svmk_PM[p]
    per2.to_excel('Non-Kmeans/Non-Kmeans_LINEAR10/Non-svmk_change'+'.xls')
if __name__ == '__main__':
    names = ['Baseline','SMOTE','RandomUnderSampler','RandomOverSampler','SOMO','TomekLinks','EditedNearestNeighbours','Borderline_SMOTE1','Safe_Level_SMOTE','SMMO','LLE_SMOTE','Stefanowski','ADASYN','SYMPROD','AHC','SMOTE_ENN']#,'ClusterCentroids']#,'AllKNN','RepeatedEditedNearestNeighbours']#,'ClusterCentroids']#,'SVM_balance']
    save_path = '不平衡比例3_data.xls'
    # data_归一化_缺失值_31.xls
    data = pd.read_excel('data_3_std.xls')#.set_index('代码')
    cluster = 10
    kmeans = False
    rd_list = [i for i in range(10,100000,10)]
    start(names,data,save_path,rd_list = rd_list,times = 100,cluster = cluster,kmeans = kmeans)
    print(cluster)
