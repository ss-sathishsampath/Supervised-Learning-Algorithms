# -*- coding: utf-8 -*-
"""
@author : Sathish Sampath(ss.sathishsampath@gmail.com)

"""

import pandas as pd
import numpy as np

# Preprocess with adult dataset
##C:\Users\sathish\Desktop\Georgia Tech\Spring 2018\ML\Assignment1
#import os
#os.chdir('Desktop/Georgia Tech/Spring 2018/ML/Assignment1')
#df = pd.read_csv('./adult.csv', header=None)
#adult = pd.read_csv('./adult2.csv',header=None)
adultf = pd.read_csv('./adult.csv', skiprows= 1)
Y = [1]*adultf.shape[0]
from sklearn.model_selection import train_test_split

adult, wast1, wast2, wast3 = train_test_split(adultf, Y, test_size=0.65, random_state=0)     


adult.columns = ['age','employer','fnlwt','edu','edu_num','marital','occupation','relationship','race','sex','cap_gain','cap_loss','hrs','country','income']
#adult['cap_gain'][5]

#adult.cap_gain.dtype = pd.np.float32
#adult['cap_gain'] = adult['cap_gain'].convert_objects(convert_numeric=True)
#adult['cap_loss'] = adult['cap_loss'].convert_objects(convert_numeric=True)

adult.describe()

#pd.to_numeric(adult, errors='coerce')
# Note that cap_gain > 0 => cap_loss = 0 and vice versa. Combine variables.
#print(adult.ix[adult.cap_gain>0].cap_loss.abs().max())
#print(adult.ix[adult.cap_loss>0].cap_gain.abs().max())
adult['cap_gain_loss'] = adult['cap_gain']-adult['cap_loss']
adult = adult.drop(['fnlwt','edu','cap_gain','cap_loss'],1)
adult['income'] = pd.get_dummies(adult.income)
print(adult.groupby('occupation')['occupation'].count())
print(adult.groupby('country').country.count())
#http://scg.sdsu.edu/dataset-adult_r/
replacements = { 'Cambodia':' SE-Asia',
                'Canada':' British-Commonwealth',
                'China':' China',
                'Columbia':' South-America',
                'Cuba':' Other',
                'Dominican-Republic':' Latin-America',
                'Ecuador':' South-America',
                'El-Salvador':' South-America ',
                'England':' British-Commonwealth',
                'France':' Euro_1',
                'Germany':' Euro_1',
                'Greece':' Euro_2',
                'Guatemala':' Latin-America',
                'Haiti':' Latin-America',
                'Holand-Netherlands':' Euro_1',
                'Honduras':' Latin-America',
                'Hong':' China',
                'Hungary':' Euro_2',
                'India':' British-Commonwealth',
                'Iran':' Other',
                'Ireland':' British-Commonwealth',
                'Italy':' Euro_1',
                'Jamaica':' Latin-America',
                'Japan':' Other',
                'Laos':' SE-Asia',
                'Mexico':' Latin-America',
                'Nicaragua':' Latin-America',
                'Outlying-US(Guam-USVI-etc)':' Latin-America',
                'Peru':' South-America',
                'Philippines':' SE-Asia',
                'Poland':' Euro_2',
                'Portugal':' Euro_2',
                'Puerto-Rico':' Latin-America',
                'Scotland':' British-Commonwealth',
                'South':' Euro_2',
                'Taiwan':' China',
                'Thailand':' SE-Asia',
                'Trinadad&Tobago':' Latin-America',
                'United-States':' United-States',
                'Vietnam':' SE-Asia',
                'Yugoslavia':' Euro_2'}
adult['country'] = adult['country'].str.strip()
adult = adult.replace(to_replace={'country':replacements,
                                  'employer':{' Without-pay': ' Never-worked'},
                                  'relationship':{' Husband': 'Spouse',' Wife':'Spouse'}})    
adult['country'] = adult['country'].str.strip()
print(adult.groupby('country').country.count())   
for col in ['employer','marital','occupation','relationship','race','sex','country']:
    adult[col] = adult[col].str.strip()
    
adult = pd.get_dummies(adult)
adult = adult.rename(columns=lambda x: x.replace('-','_'))

adult.to_hdf('datasets.hdf','adult',complib='blosc',complevel=9)

#
#wine = pd.read_csv('./wine.csv')
#wine.columns= ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","class"]
#wine.describe()
#wine.to_hdf('datasets.hdf','wine_class',complib='blosc',complevel=9)
#wine= pd.read_hdf('datasets.hdf','wine_class')  

#
#wine = pd.read_csv('./winequality-red.csv')
#wine.columns= ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
#wine.describe()
#wine.to_hdf('datasets.hdf','wine_1',complib='blosc',complevel=9)



#car = pd.get_dummies(car)

#loans = pd.read_csv('./loans.csv', skiprows= 1)
#loans.columns = ["credit_policy","purpose","int_rate","installment","log_annual_inc","dti","fico","days_with_cr_line","revol_bal","revol_util","inq_last_6mths","delinq_2yrs","pub_rec","not_fully_paid"]
#loans.describe()
#loans = pd.get_dummies(loans)
#loans.describe()
#loans = loans.dropna(how='any')
#loans.to_hdf('datasets.hdf','loans',complib='blosc',complevel=9)





biodeg = pd.read_csv('./biodeg.csv')
biodeg.columns= ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19","A20","A21","A22","A23","A24","A25","A26","A27","A28","A29","A30","A31","A32","A33","A34","A35","A36","A37","A38","A39","A40","A41","clas"]
biodeg["clas"] = pd.get_dummies(biodeg.clas)
biodeg = pd.get_dummies(biodeg)
biodeg.describe()
biodeg.to_hdf('datasets.hdf','biodeg',complib='blosc',complevel=9)




#car= pd.read_hdf('datasets.hdf','car_class') 

#abl = pd.read_csv('./abalone.csv')
#abl.columns= ["sex","length","diameter","height","whole weight","shuckled weight","viscera weight","shell weight","rings"]
#abl = pd.get_dummies(abl)
##abl = pd.get_dummies(abl, columns=["buying","maint","doors","persons","lug_boot","safety"])
#abl.describe()
##

## Madelon
#madX1 = pd.read_csv('./madelon_train.data',header=None,sep=' ')
#madX2 = pd.read_csv('./madelon_valid.data',header=None,sep=' ')
#madX = pd.concat([madX1,madX2],0).astype(float)
#madY1 = pd.read_csv('./madelon_train.labels',header=None,sep=' ')
#madY2 = pd.read_csv('./madelon_valid.labels',header=None,sep=' ')
#madY = pd.concat([madY1,madY2],0)
#madY.columns = ['Class']
#mad = pd.concat([madX,madY],1)
#mad = mad.dropna(axis=1,how='all')
#mad.to_hdf('datasets.hdf','madelon',complib='blosc',complevel=9)
