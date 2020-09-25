#!/usr/bin/env python
# coding: utf-8

# In[109]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import pandas as pd
import numpy as np



# In[110]:


pd.set_option('display.max_rows',300)


# In[111]:


df=pd.read_csv('train_loan.csv')


# In[112]:


df.head()


# In[113]:


dftest=pd.read_csv('test_loan.csv')


# In[114]:


dftest.head()


# In[115]:


df.shape


# In[116]:


df.columns


# In[117]:


df.dtypes


# In[118]:


sns.countplot(df['Loan_Status'])


# In[119]:


df.Loan_Status.value_counts()


# In[120]:


df.Loan_Status.value_counts(normalize=True)


# In[121]:


plt.figure(1)
plt.subplot(221)
df.Gender.value_counts(normalize=True).plot.bar(figsize=(20,10),title='Gender')
plt.subplot(222)
df.Married.value_counts(normalize=True).plot.bar(figsize=(20,10),title='Married')
plt.subplot(223)
df.Credit_History.value_counts(normalize=True).plot.bar(figsize=(20,10),title='Credit_History')
plt.subplot(224)
df.Self_Employed.value_counts(normalize=True).plot.bar(figsize=(20,10),title='Self_Employed')


# In[122]:


# So 80%+ people have reliable credit history.
# 80% of them are male(!!), 65% are married and 80% + self- employed


# In[123]:


df.columns


# In[124]:


plt.figure(1)
plt.subplot(131)
df.Property_Area.value_counts(normalize=True).plot.bar(figsize=(24,6),title='Property_Area')
plt.subplot(132)
df.Education.value_counts(normalize=True).plot.bar(figsize=(24,6),title='Education')
plt.subplot(133)
df.Dependents.value_counts(normalize=True).plot.bar(figsize=(24,6),title='Dependents')


# In[125]:


# So 60%+ people have no dependents, almost 80% are graduate and majortiy lives in semi- urban areas.


# In[126]:


sns.distplot(df.ApplicantIncome)  # does it represent power law distribution???


# In[127]:


sns.boxplot(df.ApplicantIncome)  #kaafi outliers


# In[128]:


df.ApplicantIncome.describe()  #monthly income


# In[129]:


df.CoapplicantIncome.describe()  # mean,std,max is much less compared to the applicant


# In[130]:


sns.distplot(df.CoapplicantIncome)  # bimodal


# In[131]:


sns.boxplot(df.CoapplicantIncome)


# In[132]:


df['LoanAmount'].isnull().sum()


# In[133]:


da=df[~df['LoanAmount'].isnull()]


# In[134]:


da['LoanAmount'].isnull().sum()


# In[135]:


sns.distplot(da['LoanAmount'])


# In[136]:


sns.boxplot(da['LoanAmount'])


# In[137]:


df.Loan_Amount_Term.describe() # thats huge! 40 saal ka loan


# In[138]:


df.Loan_Amount_Term.value_counts()


# In[139]:


# H1- is there any relation to approval with gender


# In[140]:


Gender= pd.crosstab(df.Gender,df.Loan_Status)


# In[141]:


Gender


# In[142]:


Gender['Ratio']=Gender.N/Gender.Y


# In[143]:


Gender   # almost similar


# In[144]:


# H2 - Married v/s Target


# In[145]:


Married= pd.crosstab(df.Married,df.Loan_Status)


# In[146]:


Married['Ratio']=Married.Y/Married.N


# In[147]:


Married  # for married people, the approval rate is a bit higher


# In[148]:


Dependents= pd.crosstab(df.Dependents,df.Loan_Status)


# In[149]:


Dependents['Ratio']=Dependents.Y/Dependents.N


# In[150]:


Dependents  


# In[151]:


# ek function bana lena chahiye tha


# In[152]:


Education= pd.crosstab(df.Education,df.Loan_Status)


# In[153]:


Education['Ratio']=Education.Y/Education.N


# In[154]:


Education


# In[155]:


Self_Employed= pd.crosstab(df.Self_Employed,df.Loan_Status)


# In[156]:


Self_Employed['Ratio']=Self_Employed.Y/Self_Employed.N


# In[157]:


Self_Employed  # almost same


# In[158]:



# so married, graduate , for 2 depndents it seems to be a bit higher but I guess kuch aur reason ho sakta hae.


# In[159]:


Credit_History= pd.crosstab(df.Credit_History,df.Loan_Status)


# In[160]:


Credit_History['Ratio']=Credit_History.Y/Credit_History.N


# In[161]:


Credit_History  #ofcourse


# In[162]:


Property_Area= pd.crosstab(df.Property_Area,df.Loan_Status)


# In[163]:


Property_Area['Ratio']=Property_Area.Y/Property_Area.N


# In[164]:


Property_Area # it seem to be higher for semi-urban. May be loan term is a confounding variable here.


# In[165]:


df.groupby('Loan_Status')['ApplicantIncome'].mean()  # its almost same


# In[166]:


df.ApplicantIncome.describe()


# In[167]:


df['Income_Group']=pd.cut(df.ApplicantIncome,bins=[0,2500,4000,6000,81000],labels=['Low','Average','High','VeryHigh'])


# In[168]:


Income_Group=pd.crosstab(df.Income_Group,df.Loan_Status)


# In[169]:


Income_Group['Ratio']=Income_Group.Y/Income_Group.N


# In[170]:


Income_Group # not very distinct so income grp is not a dictating factor


# In[171]:


df.LoanAmount.describe()


# In[172]:


df['Loan_Grp']=pd.cut(df.LoanAmount,bins=[0,100,200,700],labels=['Low','Average','High'])


# In[173]:


Loan_Grp=pd.crosstab(df.Loan_Grp,df.Loan_Status)


# In[174]:


Loan_Grp['Ratio']=Loan_Grp.Y/Loan_Grp.N


# In[175]:


Loan_Grp # so obviously its low for higher loan amnts


# In[176]:


df.columns


# In[177]:


df.columns


# In[179]:


del df['Income_Group']


# In[180]:


del df['Loan_Grp']


# In[189]:


matrix=df.corr()


# In[190]:


sns.heatmap(matrix,cmap="BuPu")


# In[187]:


df['Dependents'].replace('3+',3,inplace=True)
dftest['Dependents'].replace('3+',3,inplace=True)


# In[188]:


df['Loan_Status'].replace('N',0,inplace=True)
df['Loan_Status'].replace('Y',1,inplace=True)


# In[191]:


sns.heatmap(matrix,cmap="BuPu")


# In[192]:


# clearly loan status and credit history are related   Loan Amt ~ Applicant Income / Co applicant Income


# In[193]:


df.isnull().sum()


# In[ ]:


# filling gender,married,self-employed, dependents and credit history by mode


# In[200]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)


# In[201]:


dftest['Gender'].fillna(dftest['Gender'].mode()[0],inplace=True)
dftest['Married'].fillna(dftest['Married'].mode()[0],inplace=True)
dftest['Self_Employed'].fillna(dftest['Self_Employed'].mode()[0],inplace=True)
dftest['Dependents'].fillna(dftest['Dependents'].mode()[0],inplace=True)
dftest['Credit_History'].fillna(dftest['Credit_History'].mode()[0],inplace=True)


# In[202]:


df.isnull().sum()


# In[203]:


df.Loan_Amount_Term.value_counts()


# In[204]:


df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
dftest['Loan_Amount_Term'].fillna(dftest['Loan_Amount_Term'].mode()[0],inplace=True)


# In[208]:


df['LoanAmount'].fillna(df['LoanAmount'].median(),inplace=True)
dftest['LoanAmount'].fillna(dftest['LoanAmount'].median(),inplace=True)


# In[209]:


# for outlier treatment converting right skewed loan amount to log to make it a bit normal


# In[212]:


df['Log_Loan']=np.log(df['LoanAmount'])
dftest['Log_Loan']=np.log(dftest['LoanAmount'])


# In[211]:


sns.distplot(df['Log_Loan'])


# In[213]:


# Now for model building


# In[215]:


del df['Loan_ID']


# In[217]:


del dftest['Loan_ID']


# In[218]:


y=df['Loan_Status']


# In[220]:


del df['Loan_Status']


# In[221]:


df=pd.get_dummies(df)


# In[222]:


df.head()


# In[223]:


dftest=pd.get_dummies(dftest)


# In[224]:


df.columns


# In[225]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.3)


# In[226]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model=LogisticRegression()
model.fit(x_train,y_train)


# In[227]:


pred_cv=model.predict(x_train)


# In[228]:


accuracy_score(y_train,pred_cv)


# In[229]:


finaldf=pd.read_csv('sample_submission.csv')


# In[230]:


pred_test=model.predict(dftest)


# In[231]:


finaldf['Loan_Status']=pred_test


# In[232]:


finaldf.head()


# In[236]:


finaldf['Loan_Status'].replace(0,'N',inplace=True)
finaldf['Loan_Status'].replace(1,'Y',inplace=True)


# In[237]:


pd.DataFrame(finaldf,columns=['Loan_ID','Loan_Status']).to_csv('solution.csv')


# In[244]:


finaldf.to_csv('solution1.csv',index=False)


# In[ ]:


# 0.78472 accuracy for the above solution


# In[246]:


from sklearn.model_selection import StratifiedKFold


# In[248]:


kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)

i=1

for train_index,test_index in kf.split(df,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr=df.loc[train_index]
    xvl=df.loc[test_index]
    ytr=y.loc[train_index]
    yvl=y.loc[test_index]
    
    model=LogisticRegression(random_state=1)
    model.fit(xtr,ytr)
    
    pred_test=model.predict(xvl)
    
    score=accuracy_score(yvl,pred_test)
    print('accuracy score',score)
    i=i+1


# In[250]:


pred_test=model.predict(dftest)


# In[251]:


finaldf1=finaldf.copy()


# In[252]:


finaldf1['Loan_Status']=pred_test


# In[253]:


finaldf1.to_csv('solution2.csv',index=False)


# In[ ]:


# got 0.77 accuracy against test submission

