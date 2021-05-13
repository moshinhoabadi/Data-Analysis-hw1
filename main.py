#!/usr/bin/env python
# coding: utf-8


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from pycaret.regression import *


# In[3]:


def open_col(df, col_name):
    col_list = (df[col_name]
                .apply(lambda x: x if x == x else '[]')
                .apply(eval)
                .explode()
                .apply(lambda x: x if x == x else {})
               )
    col_df = pd.DataFrame(col_list.tolist(), index=col_list.index)
    return df[['movie_id']].join(col_df, how='right')
 
def month_replace(m):
    if m in ['06', '07']:
        return 'summer'
    elif m in ['11', '12']:
        return 'winter'
    else:
        return 'other'
    
def year_replace(y):
    if y <= 1960:
        return '(-inf, 1960]'
    elif y <= 1980:
        return '(1960, 1980]'
    elif y <= 1995:
        return '(1980, 1995]'
    elif y <= 2010:
        return '(1995, 2010]'
    else:
        return '(2010, inf]'

def multiple_values(df, sub_df, col_name, prefix):
    n = (sub_df
         .fillna('Other')
         .groupby('movie_id')[col_name].apply(', '.join)
         .str.get_dummies(', '))
    n.columns = prefix + n.columns
    n = n.reset_index()
    df = df.merge(n, on='movie_id')
    return df
    
def size_value(df, sub_df, prefix):
    n = sub_df.groupby('movie_id').size().reset_index()
    n = n.rename(columns={0: prefix+'_size'})
    return df.merge(n, on='movie_id')


# In[4]:


def pre_procces(file_name, train=False):

    # reading data #
    df = pd.read_csv(file_name, sep='\t')
    remove_cols = ['status', 'poster_path', 'backdrop_path', 'video']
    df = df.drop(remove_cols, axis='columns')
    df = df.rename(columns={'id':'movie_id'})

    df = df.fillna({'belongs_to_collection': ''})
    df['belongs_to_collection'] = '[' + df['belongs_to_collection'] + ']'

    subs_cols = ['belongs_to_collection', 'production_companies', 'production_countries', 
                 'genres', 'spoken_languages', 'Keywords', 'cast', 'crew']
    subs_dict = {}
    for n in subs_cols:
        sub_df = open_col(df, n)
        subs_dict[n] = sub_df

    languages = ['en', 'fr', 'es', 'de', 'it', 'ru', 'ja', 'hi', 'zh', 'ar', 'pt', 'ko', 'cn', 'la', 'pl']
    subs_dict['spoken_languages'].loc[~subs_dict['spoken_languages']['iso_639_1'].isin(languages), 'iso_639_1'] = 'Other'

    prod_comp = ['Warner Bros. Pictures', 'Universal Pictures', 'Paramount', 'Columbia Pictures', '20th Century Fox', 
                 'Metro-Goldwyn-Mayer', 'New Line Cinema', 'Canal+', 'Touchstone Pictures', 'Walt Disney Pictures', 
                 'Miramax', 'Sony Pictures', 'United Artists', 'Relativity Media', 'DreamWorks Pictures', 
                 'TriStar Pictures', 'Lionsgate', 'StudioCanal', 'Village Roadshow Pictures', 'Working Title Films',
                 'Amblin Entertainment', 'Regency Enterprises', 'Fox Searchlight Pictures', 'Focus Features', 
                 'Imagine Entertainment', 'BBC Films', 'Dimension Films', 'Film4 Productions', 'Castle Rock Entertainment', 
                 'Screen Gems', 'Hollywood Pictures', 'Dune Entertainment', 'Malpaso Productions', 'New Regency Pictures', 
                 'PolyGram Filmed Entertainment', 'Participant Media', "Centre national du cinéma et de l'image animée", 
                 'Legendary Entertainment', 'Davis Entertainment', 'TF1 Films Production']
    subs_dict['production_companies'].loc[~subs_dict['production_companies']['name'].isin(prod_comp), 'name'] = 'Other'

    prod_cntr = ['United States of America', 'United Kingdom', 'France', 'Germany', 'Canada', 'India', 'Japan', 
                 'Italy', 'Spain', 'Australia', 'China', 'Russia', 'Hong Kong', 'South Korea', 'Belgium', 'Ireland', 
                 'Sweden', 'Denmark', 'Mexico', 'Netherlands']
    subs_dict['production_countries'].loc[~subs_dict['production_countries']['name'].isin(prod_cntr), 'name'] = 'Other'


    # features_engineering #
    df['budget0'] = df['budget']==0

    if train:
        df['ratio'] = df['budget'] / df['revenue']
        df = df[(df['budget0']) | (df['ratio']>0.001)]
        df = df[df['ratio']<100]
        

    df['is_collection'] = df['belongs_to_collection'] != '[]'

    df['month'] = df['release_date'].str[5:7]
    df['month_cat'] = df['month'].apply(month_replace)
    
    df['year'] = df['release_date'].str[:4].astype('int')
    df['year_cat'] = df['year'].apply(year_replace)
    
    
    df = multiple_values(df, subs_dict['genres'], 'name', 'genre_')

    df = multiple_values(df, subs_dict['spoken_languages'], 'iso_639_1', 'lang_')
    
    df = multiple_values(df, subs_dict['production_companies'], 'name', 'prod_comp_')

    df = multiple_values(df, subs_dict['production_countries'], 'name', 'prod_cntr_')
    
    df = size_value(df, subs_dict['cast'], 'cast')
    
    df = size_value(df, subs_dict['crew'], 'crew')
    
    return df

"""

train_df = pre_procces('train.tsv', train=True)

test_df = pre_procces('test.tsv')

### predict positive numbers!!


# In[7]:


con_features = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count', 'cast_size', 'crew_size']
mlt_prefixes = ['genre', 'lang', 'prod_comp', 'prod_cntr']
mlt_features = [c for c in train_df.columns for p in mlt_prefixes if c.startswith(p+'_')]
cat_features = ['is_collection', 'month_cat', 'year_cat', 'budget0']

selected = con_features + mlt_features + cat_features
print(len(selected))

reg1 = setup(data=train_df[selected], test_data=test_df[selected], target='revenue',
             normalize=True, normalize_method='robust')


# In[ ]:





# In[20]:


compare_models(sort='rmsle')


# In[7]:


knn_model = create_model('knn')
predict_model(knn_model)
knn_model


# In[9]:


knn_model = tune_model(knn_model, optimize='rmsle', search_library='optuna', search_algorithm='tpe')
predict_model(knn_model)
knn_model


# In[ ]:


# before
# KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
#                     metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
#                     weights='uniform')
# validation 2.3275
# prediction 2.2633


# random
# baysian
# KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='manhattan',
#                     metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
#                     weights='distance')
# validation 2.2892
# prediction 2.2388

# tpe
# KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
#                     metric_params=None, n_jobs=-1, n_neighbors=23, p=2,
#                     weights='distance')
# validation 2.3902
# prediction 2.3320


# In[ ]:





# # Extra Tree

# In[10]:


et_model = create_model('et', bootstrap=False, ccp_alpha=0.0, criterion='mae',
                    max_depth=10, max_features=0.7565906156347971,
                    max_leaf_nodes=None, max_samples=None,
                    min_impurity_decrease=0.007191975016434702,
                    min_impurity_split=None, min_samples_leaf=2,
                    min_samples_split=10, min_weight_fraction_leaf=0.0,
                    n_estimators=230, n_jobs=-1, oob_score=False,
                    random_state=5331, warm_start=False)
predict_model(et_model)
et_model


# In[12]:


# et_model = tune_model(et_model, optimize='rmsle', search_library='optuna', search_algorithm='tpe')
# et_model = tune_model(et_model, optimize='rmsle', tuner_verbose=100)
et_model = tune_model(et_model, optimize='rmsle', search_library='tune-sklearn', search_algorithm='bayesian')
predict_model(et_model)
et_model


# In[ ]:


# before
# ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
#                     max_depth=None, max_features='auto', max_leaf_nodes=None,
#                     max_samples=None, min_impurity_decrease=0.0,
#                     min_impurity_split=None, min_samples_leaf=1,
#                     min_samples_split=2, min_weight_fraction_leaf=0.0,
#                     n_estimators=100, n_jobs=-1, oob_score=False,
#                     random_state=8697, verbose=0, warm_start=False)
# validation 2.2826
# prediction 2.2355



# random
# ExtraTreesRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mae', 
#                     max_depth=6, max_features=1.0, max_leaf_nodes=None, 
#                     max_samples=None, min_impurity_decrease=0.0005, 
#                     min_impurity_split=None, min_samples_leaf=3, 
#                     min_samples_split=10, min_weight_fraction_leaf=0.0, 
#                     n_estimators=280, n_jobs=-1,
#                     oob_score=False, 
#                     random_state=5331, verbose=0,
#                     warm_start=False)
# validation 2.2946
# prediction 2.2400


# baysian
# ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mae',
#                     max_depth=10, max_features=0.7565906156347971,
#                     max_leaf_nodes=None, max_samples=None,
#                     min_impurity_decrease=0.007191975016434702,
#                     min_impurity_split=None, min_samples_leaf=2,
#                     min_samples_split=10, min_weight_fraction_leaf=0.0,
#                     n_estimators=230, n_jobs=-1, oob_score=False,
#                     random_state=5331, verbose=0, warm_start=False)
# validation 2.1799
# prediction 2.1362


# tpe
# ExtraTreesRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse', max_depth=9,
#                     max_features=0.9666163614098705, max_leaf_nodes=None,
#                     max_samples=None,
#                     min_impurity_decrease=1.5993663974787875e-08,
#                     min_impurity_split=None, min_samples_leaf=1,
#                     min_samples_split=10, min_weight_fraction_leaf=0.0,
#                     n_estimators=97, n_jobs=-1, oob_score=False,
#                     random_state=5331, verbose=0, warm_start=False)
# validation 2.6104
# prediction 2.5576


# In[ ]:





# # Elastic Net

# In[13]:


en_model = create_model('en')
predict_model(en_model)
en_model


# In[16]:


# en_model = tune_model(en_model, optimize='rmsle', search_library='optuna', search_algorithm='tpe')
# en_model = tune_model(en_model, optimize='rmsle', search_library='tune-sklearn', search_algorithm='bayesian')
en_model = tune_model(en_model, optimize='rmsle', tuner_verbose=100)
predict_model(en_model)
en_model


# In[13]:


# before
# ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
#            max_iter=1000, normalize=False, positive=False, precompute=False,
#            random_state=8697, selection='cyclic', tol=0.0001, warm_start=False)
# validation 2.6268
# prediction 2.5227


# random
# ElasticNet(alpha=2.44, copy_X=True, fit_intercept=False, l1_ratio=0.512,
#            max_iter=1000, normalize=True, positive=False, precompute=False,
#            random_state=5331, selection='cyclic', tol=0.0001, warm_start=False)
# validation 2.6084
# prediction 2.5503

# baysian
# ElasticNet(alpha=0.7727239844876345, copy_X=True, fit_intercept=False,
#            l1_ratio=0.19723604623903962, max_iter=1000, normalize=True,
#            positive=False, precompute=False, random_state=5331,
#            selection='cyclic', tol=0.0001, warm_start=False)
# validation 2.6137
# prediction 2.5300

# tpe
# ElasticNet(alpha=0.994136515608409, copy_X=True, fit_intercept=False,
#            l1_ratio=0.15777391156322604, max_iter=1000, normalize=False,
#            positive=False, precompute=False, random_state=5331,
#            selection='cyclic', tol=0.0001, warm_start=False)
# validation 2.6144
# prediction 2.5574


# In[ ]:





# In[ ]:





# # Saving

# In[11]:


save_model(et_model, 'model')


# In[50]:


prediction_df = pre_procces('test.tsv')

model = load_model('model')

prediction_df = predict_model(model, prediction_df)


# In[51]:


(prediction_df['Label']<0).sum()


# In[52]:


prediction_df.loc[prediction_df['Label']<0, 'Label'] = 0


# In[53]:


(prediction_df[['movie_id', 'Label']]['Label']<0).sum()


# In[ ]:




"""