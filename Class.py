import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency

np.random.seed(31415)
#This is an attempt to create a class object which contains all functions you need in order to quickly prepare data for model training.
class data_preparation():
  def importer(self,filename,option1,option2):
    data=pd.read_csv(filename,option1,delimiter=option2)
    return data
  def numeriche(self,numeriche):
    df=pd.DataFrame(numeriche)
    df = df._get_numeric_data()
    vif = pd.DataFrame()
    vif["VIF Factor For Numeric"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif
  def categoriche(self,stringhe,etichette):
    Test=pd.DataFrame()
    for parola in etichette:
      Test[parola]=[]
    for el in list(stringhe):
      for el2 in list(stringhe):  
        if el!=el2:
          g,p,useless1,useless2=chi2_contingency(pd.crosstab(char[el],char[el2]))
          if p<0.05:
            l=[el,el2,g,p]
            l2=pd.DataFrame([l])
            l2.columns=Test.columns
            Test=Test.append(l2)
      return Test
  def num_imputation(self,numarray):
    imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
    numm=imp_num.fit_transform(numarray)
    return numm
  def cat_imputation(self,catarray):
    imp_cat = SimpleImputer(strategy="most_frequent")
    catt=imp_cat.fit_transform(catarray)
    return catt
  def cat_encoding(self,cattimputed):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(cattimputed)
    OneHotEncoder()
    catt=enc.transform(cattimputed)
    catt=catt.toarray()
    return catt
