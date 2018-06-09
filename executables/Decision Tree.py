# -*- coding: utf-8 -*-
"""
@author : Sathish Sampath(ss.sathishsampath@gmail.com)

"""

import sklearn.model_selection as ms
import pandas as pd

from helpers import basicResults,dtclf_pruned,makeTimingCurve

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def DTpruningVSnodes(clf,alphas,trgX,trgY,dataset):
    '''Dump table of pruning alpha vs. # of internal nodes'''
    out = {}
    for a in alphas:
        clf.set_params(**{'DT__alpha':a})
        clf.fit(trgX,trgY)
        out[a]=clf.steps[-1][-1].numNodes()
        print(dataset,a)
    out = pd.Series(out)
    out.index.name='alpha'
    out.name = 'Number of Internal Nodes'
    out.to_csv('output/DT_{}_nodecounts.csv'.format(dataset))
    
    return



    

# Load Data       
adult = pd.read_hdf('datasets.hdf','adult')        
adultX = adult.drop('income',1).copy().values
adultY = adult['income'].copy().values


#
#madelon = pd.read_hdf('datasets.hdf','madelon')        
#madelonX = madelon.drop('Class',1).copy().values
#madelonY = madelon['Class'].copy().values




adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     
#madelon_trgX, madelon_tstX, madelon_trgY, madelon_tstY = ms.train_test_split(madelonX, madelonY, test_size=0.3, random_state=0,stratify=madelonY)     

# Search for good alphas
alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]
##alphas=[0]
#pipeM = Pipeline([('Scale',StandardScaler()),
#                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
#                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
#                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
#                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
#                 ('DT',dtclf_pruned(random_state=55))])
#

pipeA = Pipeline([('Scale',StandardScaler()),                 
                 ('DT',dtclf_pruned(random_state=55))])


params = {'DT__criterion':['gini','entropy'],'DT__alpha':alphas,'DT__class_weight':['balanced']}

#madelon_clf = basicResults(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,params,'DT','madelon')        
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params,'DT','adult')        


#madelon_final_params = {'DT__alpha': -0.00031622776601683794, 'DT__class_weight': 'balanced', 'DT__criterion': 'entropy'}
#adult_final_params = {'class_weight': 'balanced', 'alpha': 0.0031622776601683794, 'criterion': 'entropy'}
#madelon_final_params = madelon_clf.best_params_
adult_final_params = adult_clf.best_params_

#pipeM.set_params(**madelon_final_params)
#makeTimingCurve(madelonX,madelonY,pipeM,'DT','madelon')
pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'DT','adult')


#DTpruningVSnodes(pipeM,alphas,madelon_trgX,madelon_trgY,'madelon')
DTpruningVSnodes(pipeA,alphas,adult_trgX,adult_trgY,'adult')



###################################################################################################
#
#
#banknote= pd.read_hdf('datasets.hdf','banknote') 
#banknote.describe()
#banknoteX = banknote.drop('clas',1).copy().values
#banknoteY = banknote['clas'].copy().values
#
#banknote_trgX, banknote_tstX, banknote_trgY, banknote_tstY = ms.train_test_split(banknoteX, banknoteY, test_size=0.7, random_state=0,stratify=banknoteY)     
#
#alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]
#
#
#pipeB = Pipeline([('Scale',StandardScaler()),                 
#                 ('DT',dtclf_pruned(random_state=55))])
#
#params = {'DT__criterion':['gini','entropy'],'DT__alpha':alphas,'DT__class_weight':['balanced']}
#banknote_clf = basicResults(pipeB,banknote_trgX,banknote_trgY,banknote_tstX,banknote_tstY,params,'DT','banknote')        
#
#banknote_final_params = banknote_clf.best_params_
#pipeB.set_params(**banknote_final_params)
#makeTimingCurve(banknoteX,banknoteY,pipeB,'DT','banknote')
#DTpruningVSnodes(pipeB,alphas,banknote_trgX,banknote_trgY,'banknote')


######################################################################################################
#
#
#
#diaret= pd.read_hdf('datasets.hdf','diaret') 
#diaret.describe()
#diaretX = diaret.drop('class',1).copy().values
#diaretY = diaret['class'].copy().values
#
#diaret_trgX, diaret_tstX, diaret_trgY, diaret_tstY = ms.train_test_split(diaretX, diaretY, test_size=0.25, random_state=55,stratify=diaretY)     
#
#alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]
#
#
#pipeD = Pipeline([('Scale',StandardScaler()),                 
#                 ('DT',dtclf_pruned(random_state=55))])
#
#params = {'DT__criterion':['gini','entropy'],'DT__alpha':alphas,'DT__class_weight':['balanced']}
#diaret_clf = basicResults(pipeD,diaret_trgX,diaret_trgY,diaret_tstX,diaret_tstY,params,'DT','diaret')        
#
#diaret_final_params = diaret_clf.best_params_
#pipeD.set_params(**diaret_final_params)
#makeTimingCurve(diaretX,diaretY,pipeD,'DT','diaret')
#DTpruningVSnodes(pipeD,alphas,diaret_trgX,diaret_trgY,'diaret')
#
#


#####################################################################################################


biodeg= pd.read_hdf('datasets.hdf','biodeg') 
biodeg.describe()
biodegX = biodeg.drop('clas',1).copy().values
biodegY = biodeg['clas'].copy().values

biodeg_trgX, biodeg_tstX, biodeg_trgY, biodeg_tstY = ms.train_test_split(biodegX, biodegY, test_size=0.3, random_state=0,stratify=biodegY)     

alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]


pipeB = Pipeline([('Scale',StandardScaler()),                 
                 ('DT',dtclf_pruned(random_state=55))])

params = {'DT__criterion':['gini','entropy'],'DT__alpha':alphas,'DT__class_weight':['balanced']}
biodeg_clf = basicResults(pipeB,biodeg_trgX,biodeg_trgY,biodeg_tstX,biodeg_tstY,params,'DT','biodeg')        

biodeg_final_params = biodeg_clf.best_params_
pipeB.set_params(**biodeg_final_params)
makeTimingCurve(biodegX,biodegY,pipeB,'DT','biodeg')
DTpruningVSnodes(pipeB,alphas,biodeg_trgX,biodeg_trgY,'biodeg')

