#!/usr/bin/env python
# coding: utf-8

# In[136]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm


# In[137]:


df = pd.read_csv('hw_aa.csv', sep = ';', index_col = 0)


# In[138]:


# experimentVariant – вариант эксперимента
# version – версия приложения
# purchase – факт покупки
df


# In[149]:


df.nunique()


# In[ ]:





# In[139]:


simulations = 1000    # размер симуляций
n_s = 1000    # количество наблюдений в подвыборке, которую мы будем доставать из оригинальной
res = []

df_s1 = df[df['experimentVariant'] == 1].purchase
df_s2 = df[df['experimentVariant'] == 0].purchase

# Запуск симуляций A/A теста
for i in tqdm(range(simulations)):
    s1 = df_s1.sample(n_s, replace = False).values    # из ранее сформированного эксп распределения s1 берем подвыборки БЕЗ ВОЗВРАЩЕНИЙ 
    s2 = df_s2.sample(n_s, replace = False).values    # из ранее сформированного эксп распределения s2 берем подвыборки БЕЗ ВОЗВРАЩЕНИЙ
    res.append(stats.ttest_ind(s1, s2, equal_var = False)[1]) # сохраняем pvalue

plt.hist(res, bins = 50)
plt.style.use('ggplot')
plt.xlabel('pvalues')
plt.ylabel('frequency')
plt.title("Histogram of ttest A/A simulations ")
plt.show()

# Проверяем, что количество ложноположительных случаев не превышает альфа
sum(np.array(res) <0.05) / simulations


# In[140]:


df_versions = df.groupby(['experimentVariant', 'version'], as_index = False) \
.agg({'purchase' : 'count'})


# In[141]:


df_versions


# In[142]:


# видно, что общее число наблюдений в разрезе версий приложения распределено достаточно равномерно по вариантам эксперимента
sns.countplot(x = 'version',  hue = 'experimentVariant', data = df)


# In[143]:


# посчитаем конверсию покупок в разрезе используемых пользователями приложений
df['count_purchase'] = df.purchase
df_versions = df.groupby(['experimentVariant', 'version'], as_index = False) \
.agg({'purchase' : 'sum', 'count_purchase' : 'count'})


# In[144]:


df_versions['conversion'] = df_versions.purchase  / df_versions.count_purchase


# In[145]:


# видно, что группа с приложением v2.8.0 из варианта 1 значимо отличается от остальных групп по конверсии
df_versions


# In[146]:


# однако доля ее в общей выборке весьма значима - более 65% наблюдений
df.version.value_counts(normalize = True)


# In[147]:


# удалим указанную группу из выборки и повторим эксперимент
df = df.query('version != "v2.8.0"')


# In[148]:


simulations = 1000    # размер симуляций
n_s = 1000    # количество наблюдений в подвыборке, которую мы будем доставать из оригинальной
res = []

df_s1 = df[df['experimentVariant'] == 1].purchase
df_s2 = df[df['experimentVariant'] == 0].purchase

# Запуск симуляций A/A теста
for i in tqdm(range(simulations)):
    s1 = df_s1.sample(n_s, replace = False).values    # из ранее сформированного эксп распределения s1 берем подвыборки БЕЗ ВОЗВРАЩЕНИЙ 
    s2 = df_s2.sample(n_s, replace = False).values    # из ранее сформированного эксп распределения s2 берем подвыборки БЕЗ ВОЗВРАЩЕНИЙ
    res.append(stats.ttest_ind(s1, s2, equal_var = False)[1]) # сохраняем pvalue

plt.hist(res, bins = 100)
plt.style.use('ggplot')
plt.xlabel('pvalues')
plt.ylabel('frequency')
plt.title("Histogram of ttest A/A simulations ")
plt.show()

# Проверяем, что количество ложноположительных случаев не превышает альфа
sum(np.array(res) <0.05) / simulations


# Первоначальный вариант сплитования (примерно равноразмерные группы пользователей с разными версиями) не дал удовлетворительной величины FPR из-за того, что не была учтена разница конверсий у пользователей версии v2.0.8 (которая составляет наибольшую часть тестовой аудитории), и в одну группу попало много пользователей с плохой конверсией. При удалении этой группы из рассматриваемой выборки FPR снизился, а значит, система сплитования улучшилась.
