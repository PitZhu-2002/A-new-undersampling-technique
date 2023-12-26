import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler,TomekLinks,EditedNearestNeighbours,RepeatedEditedNearestNeighbours,\
    AllKNN,CondensedNearestNeighbour,ClusterCentroids,NeighbourhoodCleaningRule,OneSidedSelection,NeighbourhoodCleaningRule,InstanceHardnessThreshold,NearMiss
from sklearn import tree
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE,RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from smote_variants import SMOTE_TomekLinks, SMOTE_ENN, Borderline_SMOTE1, Borderline_SMOTE2, ADASYN, AHC, LLE_SMOTE, DBSMOTE ,SMOTE_OUT,Supervised_SMOTE,\
    distance_SMOTE, SMMO, Stefanowski, Safe_Level_SMOTE,SVM_balance,AMSCO,SYMPROD,KernelADASYN,SOMO,NRAS,kmeans_SMOTE,MWMOTE,ANS,CCR,Gaussian_SMOTE,OUPS,SMOTE_PSO,CURE_SMOTE
from com.hdu.数据分析实战.Business_Analysis.Experiment.mode.Function import Function
from com.hdu.数据分析实战.Business_Analysis.Experiment.mode.支持向量机.Base_ratio import svm_undersampling
from com.hdu.数据分析实战.Business_Analysis.Experiment.mode.支持向量机.svm_undersampling_kmeans import \
    svm_undersampling_kmeans
import warnings
warnings.filterwarnings("ignore")

def operate(data,kmeans = False,cluster = 19):
    # 基分类器初始化
    #base = tree.DecisionTreeClassifier(random_state=10)
    #base = KNeighborsClassifier(n_neighbors=3)
    base = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=10,min_samples_leaf=2)
    # bootstrap == False 控制数据一样的变量
    if kmeans == False:
        svm_sam = svm_undersampling(data = data,bootstrap = False)
    else:
        svm_sam = svm_undersampling_kmeans(data = data,bootstrap = False,cluster = cluster)
    svm_sam.generate()
    base.fit(svm_sam.prime.iloc[:,:-1] , svm_sam.prime.iloc[:,-1])
    return base

def start(name,data,save_path,times=20,cluster=19):
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
    for train_index, test_index in stf_split.split(X, y):
        X_prime, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_prime, y_test = np.array(y)[train_index], np.array(y)[test_index]
        print(id)
        id = id + 1
        pool_sampling = []
        pool_classifier = []  # 下标 表示分类器的种类，对应 name 里面过采样方法产生的数据
        pool_prd = []
        pool_ppb = []

        # base_svm = operate(data.iloc[train_index],kmeans=False)

        # back1 = Function.cal_F1_AUC_Gmean(y_test=y_test, y_pre=base_svmk.predict(X_test),
        #                                             prob=Function.proba_predict_minority(base_svmk, X_test))
        # for l in range(len(svmk_PM)):
        #     svmk_PM[l].append(back1[l])
        # svmk_ACC.append(base_svmk.score(X_test, y_test))

        # 传统采样方法
        random_state = '(random_state = 1000)'
        for m in range(0, len(name)):
            #base = tree.DecisionTreeClassifier(random_state=10)
            #base = SVC(probability=True)
            #base = KNeighborsClassifier(n_neighbors=3)
            base = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=10,min_samples_leaf=2,random_state=10)
            if name[m] == 'Base':
                base.fit(X_prime, y_prime)
            elif name[m] in ('RandomOverSampler'):
                sample = eval(name[m] + random_state)
                X_sam, y_sam = sample.fit_resample(X_prime, y_prime)
                base.fit(X_sam, y_sam)
            elif name[m] in ('NearMiss'):
                sample = eval(name[m] +'()')
                X_sam, y_sam = sample.fit_resample(X_prime, y_prime)
                base.fit(X_sam, y_sam)
            elif name[m] in ('SMOTE'):
                sample = eval(name[m] + '(k_neighbors=5,random_state = 10)')
                print('haha')
                X_sam, y_sam = sample.fit_resample(X_prime, y_prime)
                base.fit(X_sam, y_sam)
            elif name[m] in ('AllKNN'):
                sample = eval(name[m] + '(n_neighbors=3)')
                X_sam, y_sam = sample.fit_resample(X_prime, y_prime)
                base.fit(X_sam, y_sam)
            elif name[m] in ('EditedNearestNeighbours'):
                sample = eval(name[m] + '(n_neighbors=3)')
                X_sam, y_sam = sample.fit_resample(X_prime, y_prime)
                base.fit(X_sam, y_sam)
            elif name[m] in ('ClusterCentroids','TomekLinks','RandomUnderSampler','EditedNearestNeighbours',
                'RepeatedEditedNearestNeighbours','CondensedNearestNeighbour','OneSidedSelection','NeighbourhoodCleaningRule','InstanceHardnessThreshold'):
                sample = eval(name[m]+'()')
                X_sam, y_sam = sample.fit_resample(X_prime,y_prime)
                base.fit(X_sam, y_sam)
            else:
                sample = eval(name[m] + random_state)
                X_sam, y_sam = sample.sample(X_prime, y_prime)
                base.fit(X_sam, y_sam)
            pool_classifier.append(base)
            pool_prd.append(base.predict(X_test))
            pool_ppb.append(Function.proba_predict_minority(base, X_test))
            back = Function.cal_F1_AUC_Gmean(
                y_test=y_test,
                y_pre=base.predict(X_test),
                prob=Function.proba_predict_minority(base, X_test)
            )
            for i in range(len(back)):
                PM_total[m][i].append(back[i])
            ACC_total[m].append(base.score(X_test, y_test))
    df = pd.DataFrame(columns=['Accuracy', 'F1', 'F2','Auc', 'G-mean', 'Recall', 'Precision'])
    df_roc = ['F1', 'F2','Auc', 'G-mean', 'Recall', 'Precision']
    #os.mkdir('new')
    # per2 = pd.DataFrame(columns=[i for i in range(times)])
    # per2.loc['Acc'] = svmk_ACC
    # for p in range(len(svmk_PM)):
    #     per2.loc[df_roc[p]] = svmk_PM[p]
    # per2.to_excel(str(cluster)+'/svmk'+str(cluster)+'.xls')
    #if os.path.exists('latest'):
    if os.path.exists('contrast_treeNew2'):
        print('存在')
    else:
        os.mkdir('contrast_treeNew2')
    for type in range(len(names)):
        per = pd.DataFrame(columns=[i for i in range(times)])
        save_name = names[type]
        per.loc['Acc'] = ACC_total[type]
        for j in range(len(PM_total[type])):
            per.loc[df_roc[j]] = PM_total[type][j]
        per.to_excel('contrast_treeNew2'+'/'+names[type]+'.xls')



    print(len(ACC_total))
    print(PM_total)
    # for i in range(0,len(name)):
    #     data = []
    #     data.append(np.mean(ACC_total[i]))
    #     for j in range(0,len(df.columns)-1):
    #             data.append(PM_total[i][j] / (times))
    #     df.loc[name[i]] = data



    #df.to_excel(save_path)


if __name__ == '__main__':
    #names = ['Base','SMOTE','RandomUnderSampler','CondensedNearestNeighbour','RepeatedEditedNearestNeighbours','AllKNN','ClusterCentroids','RandomOverSampler','SOMO','TomekLinks','EditedNearestNeighbours','Borderline_SMOTE1','Safe_Level_SMOTE','SMMO','LLE_SMOTE','Stefanowski','ADASYN','SYMPROD','SMOTE_ENN']
    #names = ['SMOTE','CondensedNearestNeighbour','RepeatedEditedNearestNeighbours','AllKNN','ClusterCentroids','SOMO','TomekLinks','EditedNearestNeighbours','Borderline_SMOTE1','Safe_Level_SMOTE','SMMO','LLE_SMOTE','Stefanowski','ADASYN','SMOTE_ENN']
    #names = ['ANS','CCR','Gaussian_SMOTE']
    #names = ['Base','MWMOTE']
    #names = ['SMOTE']#,'TomekLinks']#,'CURE_SMOTE']
    #names = ['Base']#,'RepeatedEditedNearestNeighbours']
    #names = ['ClusterCentroids']
    names = ['RandomUnderSampler']
    #names = ['Base','RandomOverSampler','RandomUnderSampler','SMOTE','CCR','kmeans_SMOTE','NRAS','ADASYN','Gaussian_SMOTE','OneSidedSelection','NeighbourhoodCleaningRule','EditedNearestNeighbours','CURE_SMOTE',
             #'SOMO','MWMOTE','ANS','Safe_Level_SMOTE','Borderline_SMOTE1','SYMPROD','RepeatedEditedNearestNeighbours','CondensedNearestNeighbour','AllKNN','ClusterCentroids']
             #'TomekLinks','NeighbourhoodCleaningRule','OneSidedSelection','InstanceHardnessThreshold','EditedNearestNeighbours','Stefanowski']
    #names = ['RepeatedEditedNearestNeighbours','CondensedNearestNeighbour','AllKNN','ClusterCentroids',
             #'TomekLinks','NeighbourhoodCleaningRule','OneSidedSelection','InstanceHardnessThreshold','EditedNearestNeighbours','Stefanowski']
    #names = ['CURE_SMOTE']
    save_path = '不平衡比例3_data.xls'
    # data_归一化_缺失值_31.xls
    data = pd.read_excel('data_4_std.xls')#.set_index('代码')
    #data.iloc[:,:-1] = scale(data.iloc[:,:-1])
    start(names,data,save_path,times = 100,cluster = 20)
