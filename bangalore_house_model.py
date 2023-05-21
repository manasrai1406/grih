#!/usr/bin/env python
# coding: utf-8

# In[156]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[157]:


df = pd.read_csv("C://Users//mrai7//OneDrive//Desktop//archive//Bengaluru_House_Data.csv")


# In[158]:


df.head()


# In[159]:


df.tail()


# In[160]:


df.columns


# In[161]:


df.corr


# In[162]:


df.shape


# In[163]:


df.info()


# In[164]:


for column in df.columns:
    print(df[column].value_counts())
    print("*"*20)


# In[165]:


df.isnull().sum()


# In[166]:


df.drop(columns=["area_type","availability","society","balcony"],inplace = True)


# In[167]:


df.describe()


# In[168]:


df.info()


# In[169]:


df['location'].value_counts()


# In[170]:


df['location']=df['location'].fillna('Sarjapur  Road')


# In[171]:


df['location'].isnull().sum()


# In[172]:


df['size'].value_counts()


# In[173]:


df.size=df['size'].fillna(" 2 BHK")


# In[174]:


df['bath']=df['bath'].fillna(df['bath'].median())


# In[175]:


df['bhk']=df['size'].str.split().str.get(0).astype(int)


# In[176]:


df[df.bhk >20]


# In[177]:


df['total_sqft'].unique()


# In[178]:


df.info()


# In[179]:


def convertrange(x):
    temp = x.split('-')
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1]))/2
    try:
        return float(x)
    except:
        return None


# In[180]:


df['total_sqft'] = df['total_sqft'].apply(convertrange)


# In[181]:


df.head()


# In[182]:


df.head()


# In[ ]:





# In[183]:


df['price_per_sqft'] = df['price']*100000/df['total_sqft']


# In[184]:


df.describe()


# In[185]:


df['location'].value_counts()


# In[186]:


df['location'] = df['location'].apply(lambda x : x.strip())
location_count = df['location'].value_counts()


# In[187]:


location_count_less_10 = location_count[location_count <= 10]
location_count_less_10


# In[188]:


df['location'] = df['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)


# In[189]:


df['location'].value_counts()


# In[190]:


(df['total_sqft']/df['bhk']).describe()


# In[191]:


df = df[((df['total_sqft']/df['bhk'])>= 300)]
df.describe()


# In[192]:


df.shape


# In[193]:


df.price_per_sqft.describe()


# In[194]:


def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st= np.std(subdf.price_per_sqft)
        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output, gen_df], ignore_index=True)
    return df_output
df= remove_outliers_sqft(df)
df.describe()


# In[195]:


def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats= {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]= {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices =np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)            
    return df.drop(exclude_indices, axis='index')


# In[196]:


df=bhk_outlier_remover(df)


# In[197]:


df


# In[198]:


df.shape


# In[199]:


df.drop(columns=['size', 'price_per_sqft'], inplace=True)


# In[200]:


df.columns


# In[201]:


X=df.drop(columns=['price'])
y=df['price']


# In[202]:


df.info()


# In[203]:


df.head()


# In[204]:


df.info()


# In[205]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[213]:


X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=0)
column_trans =make_column_transformer((OneHotEncoder(sparse=False), ['location']), remainder='passthrough')


# In[214]:


scaler = StandardScaler()


# In[215]:


lr = LinearRegression()


# In[216]:


pipe =make_pipeline(column_trans, scaler, lr)


# In[217]:


pipe.fit(X_train, y_train)


# In[ ]:





# In[218]:


y_pred_lr = pipe.predict(X_test)


# In[231]:


print(y_pred_lr)


# In[ ]:





# In[ ]:




