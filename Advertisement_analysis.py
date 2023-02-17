#!/usr/bin/env python
# coding: utf-8

# # Данные:
#     взаимодействий с **рекламными объявлениями** на некоторой площадке за 6 дней:
#     Описание данных
# ads_data.csv – данные взаимодействий с рекламными объявлениями  
# date – дата  
# time – время  
# event – действие (просмотр/клик)  
# platform – платформа  
# ad_id – id объявления  
# client_union_id – id рекламного кабинета  
# campaign_union_id – id рекламной кампании  
# ad_cost_type – тип оплаты  
# ad_cost – цена  
# has_video – есть ли видео  
# target_audience_count – размер аудитории  
# 
#     таблица с характеристиками **рекламных клиентов** (тех, кто разместил эти объявления):  
# ads_clients_data.csv – характеристики рекламных клиентов  
# date – дата  
# client_union_id – id рекламного кабинета  
# community_id – id сообщества  
# create_date – дата создания рекламного клиента  
# 
# ## Задачи
# ### датафрейм объявлений
# 1. Среднее количество показов и среднее количество кликов на объявления за весь период (округлить до целых).
# 2. График распределения показов на объявление за весь период.
# 3. Посчитать скользящее среднее показов с окном 2. 
# 4. Скользящее среднее часто используется для поиска аномалий в данных. Нанести на один график значения арифметического среднего по дням и скользящего среднего количества показов. В какой день наблюдается наибольшая разница по модулю между арифметическим средним и скользящим средним? Дни, в которых скользящее среднее равно NaN, не учитываем. 
# 5. Написать функцию, которая найдет проблемное объявление (с наибольшим/наименьшим количеством показов) в день, в который была замечена самая большая по модулю аномалия. 
# ### датафрейм клиентов
# 1. Найти среднее количество дней от даты создания рекламного клиента и первым запуском рекламного объявления этим клиентом.
# 2. Вычислить конверсию из создания рекламного клиента в запуск первой рекламы в течение не более 365 дней (ответ в процентах и округлить до сотых). 
# 3. Разбить наших клиентов по промежуткам от создания до запуска рекламного объявления, равным 30. Определить, сколько уникальных клиентов запустили свое первое объявление в первый месяц своего существования (от 0 до 30 дней). Список промежутков  – [0, 30, 90, 180, 365]

# In[17]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import plotly.express as px
import os


# In[18]:


# задаем формат отображения графиков
sns.set(
    font_scale=2,
    style="whitegrid",
    rc={'figure.figsize':(20,7)}
        )


# In[19]:


# считываем данные об объявлениях
ads_data = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-e-malofeeva-22/shared/homeworks/python_ds_miniprojects/6/ads_data.csv')


# In[20]:


ads_data.head()


# In[21]:


ads_data.dtypes


# In[22]:


ads_data.groupby(['ad_cost_type', 'event']) \
.agg({'platform' : 'count'})


# In[23]:


# Среднее количество показов и среднее количество кликов на объявления за весь период (округлить до целых):
ads_data.groupby(['ad_id', 'event']) \
.agg({'platform' : 'count'}) \
.unstack() \
.fillna(0) \
.mean() \
.round()


# In[24]:


# График распределения показов на объявление за весь период:
ads_data_view = ads_data.query('event == "view"') \
.groupby('ad_id', as_index = False) \
.agg({'platform' : 'count'}) \
.rename(columns = {'platform' : 'view_count'})


# In[25]:


ads_data_view.head()


# In[26]:


ads_data_view['log_view'] = np.log(ads_data_view.view_count)


# In[27]:


ads_data_view.head()


# In[28]:


sns.distplot(ads_data_view.log_view)


# In[29]:


# Посчитать скользящее среднее показов с окном 2. 
# Какое значение скользящего среднего получим за 6 апреля 2019 года (ответ до целых):
ads_data.query('event == "view"') \
.groupby(['ad_id', 'date']) \
.agg({'platform' : 'count'}) \
.unstack() \
.mean() \
.rolling(2).mean() \
.round()


# In[30]:


# Нанести на один график значения арифметического среднего по дням и скользящего среднего количества показов. 
# В какой день наблюдается наибольшая разница по модулю между арифметическим средним и скользящим средним? 
# Дни, в которых скользящее среднее равно NaN, не учитываем:
rolling_data = ads_data.query('event == "view"') \
.groupby(['ad_id', 'date']) \
.agg({'platform' : 'count'}) \
.unstack() \
.mean() \
.rolling(2).mean()


# In[34]:


rolling_data = rolling_data.reset_index() \
.rename(columns = {0 : 'mean'})


# In[35]:


rolling_data


# In[39]:


del rolling_data['level_0']


# In[40]:


rolling_data = rolling_data[rolling_data['mean'] == rolling_data['mean']]


# In[41]:


rolling_data


# In[44]:


mean_data = ads_data.query('event == "view"') \
.groupby(['ad_id', 'date']) \
.agg({'platform' : 'count'}) \
.unstack() \
.mean()


# In[46]:


mean_data = mean_data.reset_index().rename(columns = {0 : 'mean'})


# In[48]:


del mean_data['level_0']


# In[49]:


sns.lineplot(data = mean_data, x = 'date', y = 'mean')
sns.lineplot(data = rolling_data, x = 'date', y = 'mean')


# In[50]:


means_data = rolling_data.merge(mean_data, on = 'date') \
.rename(columns = {'mean_x' : 'rolling_mean', 'mean_y' : 'mean'})


# In[51]:


means_data


# In[52]:


means_data['abs_diff'] = means_data['rolling_mean'] - means_data['mean']


# In[53]:


means_data.abs_diff = means_data.abs_diff.apply(lambda x: abs(x))


# In[54]:


means_data


# In[ ]:


# Написать функцию, которая найдет проблемное объявление (с наибольшим/наименьшим количеством показов) в день, 
# в который была замечена самая большая по модулю аномалия.


# In[55]:


# считываем данные о клиентах
ads_clients_data = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-e-malofeeva-22/shared/homeworks/python_ds_miniprojects/6/ads_clients_data.csv')


# In[56]:


ads_clients_data.head()


# In[57]:


ads_clients_data.dtypes


# In[58]:


ads_clients_data.shape


# In[59]:


# из-за сходных наименований переименуем колонку в рекламном датафрейме
ads_data = ads_data.rename(columns = {'date' : 'date_ads'})


# In[60]:


# объединим данные рекламы с данными о рекламных клиентах:
ads_data_union = ads_data.merge(ads_clients_data, on = 'client_union_id')


# In[61]:


ads_data_union.columns


# In[ ]:


# найдем среднее количество дней от даты создания рекламного клиента (create_date) 
# и первым запуском рекламного объявления этим клиентом (date_ads):


# In[62]:


# переведем нужные колонки дат в формат to_datetime()
ads_data_union.date_ads = pd.to_datetime(ads_data_union.date_ads)


# In[63]:


# переведем нужные колонки дат в формат to_datetime()
ads_data_union.create_date = pd.to_datetime(ads_data_union.create_date)


# In[64]:


ads_data_union.dtypes


# In[65]:


# оценим разницу между днем создания клиента и объявления для всех объявлений
ads_data_union['diff_time'] = ads_data_union.date_ads - ads_data_union.create_date


# In[66]:


ads_data_union.head()


# In[67]:


# выберем только те объявления, которые имеют минимальный разрыв между созданием клиента и объявления (очевидно, были первыми)
# посмотрим на их среднее
ads_data_union.groupby('client_union_id', as_index = False) \
.agg({'diff_time' : 'min'}) \
.mean()


# In[ ]:


# Вычислить конверсию из создания рекламного клиента в запуск первой рекламы в течение не более 365 дней 
# (ответ в процентах и округлить до сотых). 


# In[68]:


# сколько уникальных клиентов?
ads_clients_data.client_union_id.nunique()


# In[69]:


# сколько из них создали рекламу в течение года?
ads_data_union.client_union_id.nunique()


# In[70]:


# конверсия клиентов
838 * 100/122078


# In[71]:


ads_data_union.head()


# In[ ]:


# Разбить наших клиентов по промежуткам от создания до запуска рекламного объявления, равным 30. 


# In[83]:


cut_bins = pd.to_timedelta(['0d', '30d', '90d', '180d', '365d', '106751d'])


# In[79]:


cut_labels = ['first month', 'third month', 'half year', 'year', 'more than year']


# In[85]:


ads_data_union['time_categorical'] = pd.cut(ads_data_union.diff_time, bins = cut_bins, labels = cut_labels)


# In[86]:


ads_data_union.head()


# In[102]:


# Определить, сколько уникальных клиентов запустили свое первое объявление в первый месяц своего существования (от 0 до 30 дней)
groups = ads_data_union.groupby('time_categorical', as_index = False) \
.agg({'client_union_id' : 'nunique'})


# In[103]:


groups

