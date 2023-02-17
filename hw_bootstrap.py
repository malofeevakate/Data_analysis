#!/usr/bin/env python
# coding: utf-8

# ## Бутстрап
# Бутстрап позволяет многократно извлекать подвыборки из выборки, полученной в рамках экспериментва
# 
# В полученных подвыборках считаются статистики (среднее, медиана и т.п.)
# 
# Из статистик можно получить ее распределение и взять доверительный интервал
# 
# ЦПТ, например, не позволяет строить доверительные интервал для медианы, а бутстрэп это может сделать

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
import seaborn as sns

plt.style.use('ggplot')


# In[2]:


# Объявим функцию, которая позволит проверять гипотезы с помощью бутстрапа
def get_bootstrap(
    data_column_1, # числовые значения первой выборки
    data_column_2, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
):
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1 = data_column_1.sample(
            len(data_column_1), 
            replace = True # параметр возвращения
        ).values
        
        samples_2 = data_column_2.sample(
            len(data_column_1), 
            replace = True
        ).values
        
        boot_data.append(statistic(samples_1)-statistic(samples_2)) # mean() - применяем статистику
        
    pd_boot_data = pd.DataFrame(boot_data)
        
    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])
        
    p_1 = norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_2 = norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2
        
    # Визуализация
    _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
    for bar in bars:
        if bar.get_x() <= quants.iloc[0][0] or bar.get_x() >= quants.iloc[1][0]:
            bar.set_facecolor('red')
        else: 
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')
    
    plt.style.use('ggplot')
    plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data")
    plt.show()
       
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value}


# In[3]:


# считаем данные
df = pd.read_csv('hw_bootstrap.csv', sep = ';', index_col = 0, decimal = ',')


# In[4]:


df


# In[5]:


df.dtypes


# In[6]:


# посмотрим на размеры теста и контроля
df.groupby('experimentVariant').agg({'value': 'count'})


# In[7]:


# сравним их средние
df.groupby('experimentVariant').agg({'value': 'mean'})


# In[8]:


# и медианы
df.groupby('experimentVariant').agg({'value': 'median'})


# In[9]:


# при построении боксплотов уже видно, что тестовая группа отличается большим количеством выбросов
sns.boxplot(x="experimentVariant", y="value", data=df)


# In[10]:


# распределение контрольной группы внешне похоже на нормальное
sns.distplot(df[df.experimentVariant == 'Control'].value, kde=False)


# In[11]:


# чего не скажешь о тестовой
sns.distplot(df[df.experimentVariant == 'Treatment'].value, kde=False)


# In[12]:


# Манн-Уитни не позволяет отвергнуть нулевую гипотезу об однородности выборок
mannwhitneyu(df[df.experimentVariant == 'Treatment'].value, 
             df[df.experimentVariant == 'Control'].value)


# In[13]:


# а t-test дает значимые различия
ttest_ind(df[df.experimentVariant == 'Treatment'].value, 
             df[df.experimentVariant == 'Control'].value)


# In[14]:


# посмотрим на бутстрэп по среднему
get_bootstrap(
    df[df.experimentVariant == 'Treatment'].value, # числовые значения первой выборки
    df[df.experimentVariant == 'Control'].value, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
)


# In[25]:


# посмотрим на бутстрэп по медиане показывает, что недостаточно оснований отвергать нулевую гипотезу о равенстве медиан
# то есть, у контроля и теста медианы одинаковые, но средние значимо различаются из-за выбросов в тесте
get_bootstrap(
    df[df.experimentVariant == 'Treatment'].value, # числовые значения первой выборки
    df[df.experimentVariant == 'Control'].value, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.median, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
)


# **Бутстрэп по среднему и т-тест** дали оба p_value < 0.05, что предполагает **отвержение нулевой гипотезы** об однородности выборок. **Манн-Уитни** наоборот говорит, что **недостаточно оснований отвергать нулевую гипотезу**. Но посмотрим на визуализацию наших выборок: если контроль распределен более - менее равномерно, то в тестовой группе значения в основном пляшут около 10, но имеются просто ужасающие выбросы. Логично, что t-test и бутстрэп не показали хорошие результаты, выбросы тестовой выборки смазали результаты тестов. Однако Манн-Уитни, в силу того, что ранжирование нивелирует сильные различия, показал, что выборки значимо не различаются. Попробуем убрать выбросы и повторить тесты

# In[15]:


df1 = df.iloc[0:995]


# In[16]:


# размеры теста и контроля не сильно отличаются
df1.groupby('experimentVariant').agg({'value': 'count'})


# In[17]:


# средние тоже практически одинаковые, в противовес первоначальному результату (различались вдвое)
df1.groupby('experimentVariant').agg({'value': 'mean'})


# In[18]:


# аналогично медианы
df1.groupby('experimentVariant').agg({'value': 'median'})


# In[19]:


# боксплоты демонстрируют практически аналогичность тестовой и контрольной групп
sns.boxplot(x="experimentVariant", y="value", data=df1)


# In[20]:


# тест стал практически нормально распределен
sns.distplot(df1[df1.experimentVariant == 'Treatment'].value, kde=False)


# In[21]:


# Манн-Уитни все также не позволяет отвергнуть нулевую гипотезу об однородности выборок
mannwhitneyu(df1[df1.experimentVariant == 'Treatment'].value, 
             df1[df1.experimentVariant == 'Control'].value)


# In[22]:


# а t-test также показывает, что тест и контроль одинаковы
ttest_ind(df1[df1.experimentVariant == 'Treatment'].value, 
             df1[df1.experimentVariant == 'Control'].value)


# In[23]:


# и бутстрэп дает замечательный результат
get_bootstrap(
    df1[df1.experimentVariant == 'Treatment'].value, # числовые значения первой выборки
    df1[df1.experimentVariant == 'Control'].value, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
)


# Итак, чем нормальнее распределены проверяемые выборки (и гомоскедастичны их дисперсии), тем больше подходит t-test. Манн-Уитни хорошо справляется с сильными выбросами, а бутстрап при значимых выбросах может ошибаться в оценке среднего, но давать приемлемые оценки других параметров (медиана). 
