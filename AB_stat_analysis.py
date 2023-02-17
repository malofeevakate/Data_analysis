#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import scipy.stats as ss
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.api import anova_lm
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# считаем данные о реакциях на разные форматы фото в нашем приложении
df = pd.read_csv('5_task_1.csv')


# In[4]:


df


# In[7]:


# проверим данные внутри групп на гомоскедастичность
pg.homoscedasticity(df, dv = 'events', group = 'group', method = 'levene', alpha = 0.05)


# In[10]:


# проверим данные внутри групп на нормальность
pg.normality(df, dv = 'events', method = 'normaltest', group = 'group')


# In[26]:


# так как недостаточно оснований отклонить гипотезы о нормальности данных и гомоскедастичности дисперсий, проведем дисперсионный анализ
pg.anova(data = df, dv = 'events', between = 'group')


# In[22]:


# P < 0.05, есть основания отклонить нулевую гипотезу (то есть между группами имеются значимые различия)
# сравним попарно средние, используя критерий Тьюки

pg.pairwise_tukey(df, dv = 'events', between = 'group')


# In[23]:


# p < 0,05, значит, между всеми группами имеются статзначимые различия
# построим визуализацию и выберем предпочтительный вариант

sns.pointplot(x = 'group', y = 'events', data = df, capsize = 0.2)


# In[2]:


# считаем данные о реакциях на разный вид кнопки заказа
df_button = pd.read_csv('5_task_2.csv')


# In[7]:


df_button


# требуется проверить, как пользователи отреагируют на изменение формата кнопки оформления заказа, с разбивкой по сегменту клиента. Поскольку у нас есть несколько переменных - зависимая (число событий) и два независимых фактора (сегмент клиента и его группа) - используем многофакторный ANOVA

# In[3]:


# выделим тестовые значения из датафрейма
df_button_test = df_button.query('group == "test"')


# In[4]:


# выделим контрольные занчения из датафрейма
df_button_control = df_button.query('group == "control"')


# In[5]:


# посмотрим на распределение реакций в тестовой группе
sns.distplot(df_button_test.events, kde = False)


# In[6]:


# посмотрим на распределение реакций в контрольной группе
sns.distplot(df_button_control.events, kde = False)


# In[7]:


df_button_test.query('segment == "high"').events.describe()


# In[8]:


# выведем описательные статистики для каждого сегмента в каждой группе
df_button_test.query('segment == "low"').events.describe()


# In[9]:


# выведем описательные статистики для каждого сегмента в каждой группе
df_button_control.query('segment == "high"').events.describe()


# In[10]:


# выведем описательные статистики для каждого сегмента в каждой группе
df_button_control.query('segment == "low"').events.describe()


# In[11]:


# для проведения анализа создадим доп колонку с объединением данных о группе и сегменте клиента
df_button['comb'] = df_button.group + '/' + df_button.segment


# In[12]:


df_button.head()


# In[13]:


# ANOVA говорит, что среди указанных 4 подгрупп наблюдаются значимые различия
pg.anova(df_button, dv = 'events', between = 'comb')


# In[14]:


# из-за множественных сравнений применим критерий Тьюки и определим, между какими подгруппами имеются зачимые различия
pg.pairwise_tukey(df_button, dv = 'events', between = 'comb')


# In[19]:


# визуализируем средние внутри подгрупп
sns.pointplot(x = 'comb', y = 'events', hue = 'group', data = df_button, capsize = .2)
plt.title('Распределение средних значений реакций внутри групп и сегментов клиентов')
plt.xlabel('Подгруппа клиента')
plt.ylabel('Среднее по реакциям')


# Итого, мы наблюдаем стат значимые различия между всеми подгруппами, при этом средние в тестовых подгруппах выше, чем в контрольных (в обоих клиентских сегментах).  Соответственно, имеется достаточно оснований выкатить новый дизайн кнопки на всю клиентскую базу
