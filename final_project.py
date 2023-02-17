#!/usr/bin/env python
# coding: utf-8

# # Final_project_variant2 #

# Исследуются данные о работе некоторого образовательного сервиса. Было проведено A/B тестирование разных вариантов оплаты. Необходимо:
# 1. Проанализировать результаты A/B тестирования  
# 2. С помощью SQL-запросов выяснить число лояльных (очень усердных) студентов, а также продуктовые метрики    
# 3. Написать функции обновления данных из сторонних файлов, пересчета метрик и визуализации результатов  
# 
# ## Задание 1. A/B–тестирование ##
# В ходе A/B тестирования одной гипотезы **целевой группе была предложена новая механика оплаты услуг на сайте, у контрольной группы оставалась базовая механика**. Необходимо проанализировать итоги эксперимента и сделать вывод, стоит ли запускать новую механику оплаты на всех пользователей.  
# В качестве входных данных есть 4 csv-файла:  
# - groups.csv - файл с информацией о принадлежности пользователя к контрольной или экспериментальной группе (А – контроль, B – целевая группа) 
# - groups_add.csv - дополнительный файл с пользователями, который вам прислали спустя 2 дня после передачи данных  
# - active_studs.csv - файл с информацией о пользователях, которые зашли на платформу в дни проведения эксперимента 
# - checks.csv - файл с информацией об оплатах пользователей в дни проведения эксперимента.

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from io import BytesIO
import requests
import pingouin as pg

from tqdm.auto import tqdm
import seaborn as sns

from datetime import timedelta
from datetime import datetime
from io import BytesIO

from airflow import DAG
from airflow.decorators import dag, task

plt.style.use('ggplot')


# ### Анализ входных данных ###

# In[5]:


# Посмотрим на данные о распределении тестовой и контрольной групп
req1 = requests.get('https://s08vlx.storage.yandex.net/rdisk/8b683f76025c5bf7b1badd2c6859171800ffa09b8796b2a8e713d5b7d1918709/63209342/fKqInKw3d7bLFOeFnMGnhLWIS9AmXMMXWGR17A-BkO3-qg5Wp8tUAp7lxoJJvJyG8pjltSuuxShICmGl2FDiJhbsM4PC04KdB-NoZUjfeOOr8npumZHI4midPdWhecNq?uid=0&filename=%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82_2_groups.csv&disposition=attachment&hash=FjLs%2B6AJdyaRoyFMFeL4raj9RoBP3hYIuXfguWvuEGcjdjYEIrmN7ZN/eZ32Z77Cq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=text%2Fplain&owner_uid=125175473&fsize=724984&hid=4b9098e1d545e562c779de19f9f64e2f&media_type=spreadsheet&tknv=v2&rtoken=aTE0kCXZNHIP&force_default=no&ycrid=na-d3635131a928a2c7b8c603745d05bf0a-downloader16h&ts=5e88fcef9d480&s=0e45dc55fe9181d294d5ef8d3aabedb8c6bf7bbd35db78f3c395180e30f525e7&pb=U2FsdGVkX1-6VB28x0NC8BEpsXYiVFgu6NIWKdJr1w4Z56HZIxvh_a-arAWCzrVW2oifz3ekGOIfq-C0iG-A7nCuu_3eNutzRlHA6UdqzOGsJajRGKqPL9EYxP4kT3vU')
df_groups = pd.read_csv(BytesIO(data1), sep = ';')


# In[5]:


df_groups.head(5)


# In[4]:


# пропущенные значения отсутствуют, все пользователи распределены по группам
df_groups.info()


# In[34]:


# численное распределение пользователей по группам неравномерное, очень странно, что целевая группа настолько больше контрольной
df_groups.groupby('grp', as_index = False).count()


# In[9]:


# проверим на повторы айдишников
df_groups.id.duplicated().sum()


# Имеется фрейм с уникальными айдишками 74484 юзеров, поделенных на тестовую и контрольную группы. Разделение этих групп кажется странным: раскатывать изменения на большой группе пользователей, и брать размер контроля только 25% от теста. Хотелось бы взглянуть на систему сплитования как минимум.

# In[ ]:


# рассмотрим данные о новых юзерах, присланные через 2 дня
req4 = requests.get('https://s750sas.storage.yandex.net/rdisk/f06bf3fb0ce3d7ed783c715aa18a0389af6e9d4efa00d64ca5f53aac6182abf1/62ffc01e/fKqInKw3d7bLFOeFnMGnhGAdJ2MSdW82i4_ghmwgsZlWhJkpRytn07Ls4kX484X-lezQy4loqSz9AFrIeHDnv8QEHwKSTiYbb_hzq1q6XQir8npumZHI4midPdWhecNq?uid=0&filename=%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82_2_group_add.csv&disposition=attachment&hash=0ertv33lnyTdzTu9NbdN9MtgpuTSXleSHZ%2BvhK85v05jGwJs7DHSrj4ununKsRcfq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=text%2Fplain&owner_uid=125175473&fsize=949&hid=e8dcec35987c367a56f82404a22463b7&media_type=spreadsheet&tknv=v2&rtoken=x7sQvDBvEcVd&force_default=no&ycrid=na-6ae8cf4f9feb8da91b76dd02295986e5-downloader2e&ts=5e69af139c380&s=28a8f791e2b55bce136323bbd03a635c3fae77f3b10cbcaff7b7a06d71e43fd6&pb=U2FsdGVkX18Wzaw8soN5pjC05gU8iWY8NCgpwE3Bgchzktb4Mdsjxki6tJpzFvDFmTxaIG_MODh8nBJa5LDlTyyWkkVUpW84A0_EDWJ33f5Fytt1LWGLsoVjXXtP2sIv')
data4 = req4.content
df_groups_add = pd.read_csv(BytesIO(data4))


# In[11]:


df_groups_add.head()


# In[6]:


# аналогично, пропущенные значения отсутствуют, все пользователи распределены по группам
df_groups_add.info()


# In[38]:


# численное распределение пользователей по группам: около 30% юзеров относятся к контрольной группе, и 70% - к тестовой
df_groups_add.groupby('grp', as_index = False).count()


# In[20]:


# проверим на повторы айдишников
df_groups_add.id.duplicated().sum()


# Аналогично основной группе, добавочная оказалась из уникальных айдишников 92 юзеров, все поделены на тест и контроль. Соотношение деления все такое же странное. Учитывая, что изменение/доработка системы оплаты сколько-то стоит (в отличие от отсутствия изменений как таковых), можно подумать, что описание датафрейма неверно, и А - группа тестовая, а В - контрольная. Ну или тест на самом деле ухудшающий (правда, не могу придумать пример в рамках системы оплаты). В любом случае, **сплитование вызывает вопросы**.

# In[5]:


# получим общий итоговый фрейм по распределению юзеров
df_groups_total = pd.concat([df_groups, df_groups_add])


# In[8]:


df_groups_total.head()


# In[6]:


# видим, что размер общего фрейма как раз равен сумме строчек df_groups и df_groups_add, все ок
df_groups_total.info()


# In[7]:


# переименуем колонку с айдишниками пользователей для удобного использования
df_groups_total = df_groups_total.rename(columns={'id' : 'student_id'})


# In[47]:


# все айдишники уникальны
df_groups_total.student_id.nunique()


# In[111]:


# посмотрим на активных пользователей в дни эксперимента
req2 = requests.get('https://s432vla.storage.yandex.net/rdisk/3a40f377f696432a2588a986b2af49044dd68afb5b18d69010f8282c1f40f146/62ffbec8/n3N0SQYM2SoDHEJIQgKt2ujEDw5jYN17sSz14UE4iwGyYlrHRmnPoV9WI8mqa1dmF6Qt_88ffG90wBmmEpDhFQ==?uid=0&filename=%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82_2_active_studs.csv&disposition=attachment&hash=zWS0uLxK3MRXd/4/IErmDFQBGvbpGD1qm/aJx/k6EUUtE6DaDI0uGkDT4cHh/WJbq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=text%2Fplain&owner_uid=125175473&fsize=65131&hid=3ed23002f82477fd8c7d4a21c17d8034&media_type=spreadsheet&tknv=v2&rtoken=VLQnNMBdZHav&force_default=no&ycrid=na-96300cdc146eb8e9f9841f7d1c03ae4f-downloader11f&ts=5e69adcd74200&s=298f3f9a6a0b4dcfcd30b5e2531375074e1817b4485c4e2acc2cdda94c064471&pb=U2FsdGVkX19nAru0O76sUeX5I7sjU7s3OfVwfBSgj3bVELERwhe44uxz5-Qbg-huJf5HKtETRz_t9FsHXfXWWfoU1L7utUUpsCWTRvpOBWo')
data2 = req2.content
df_active_studs = pd.read_csv(BytesIO(data2))


# In[12]:


# мы видим перечень юзеров, которые заходили на платформу в дни эксперимента
df_active_studs.head(5)


# In[45]:


# 8341 юзер заходил на платформу в дни эксперимента
df_active_studs.shape


# In[13]:


# все айдишники уникальны
df_active_studs.student_id.nunique()


# In[9]:


# для удобства добавим колонку с отметкой об активности юзеров
df_active_studs['activity'] = 'yes'


# In[115]:


# посмотрим данные об оплате
req3 = requests.get('https://s251vla.storage.yandex.net/rdisk/2ed0cef7d3eeb388a1cfef16c898d547abd95ff3590fc483208d4d0329f39e40/62ffbf91/n3N0SQYM2SoDHEJIQgKt2psRrG10OKyAasqgpda4iX7NDTSjoP6nmHZtzOqJXgQH436kYOx6cWVM8zgvGq_zHg==?uid=0&filename=%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82_2_checks.csv&disposition=attachment&hash=TeGedpkC9yRvIkaaNDf1zC3bFkr2Kvo9MvW7kYjfHc5svktv83UZ47kn6%2BaauC1pq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=text%2Fplain&owner_uid=125175473&fsize=7764&hid=caf94adff3374e1c1a09fba3aae11e50&media_type=spreadsheet&tknv=v2&rtoken=lCY7FQdHOPMD&force_default=no&ycrid=na-5f259d0cfa6acec9fbf8a1703b68f794-downloader2e&ts=5e69ae8d24640&s=5e3efcf73091f37afe750f8ddb7beb1773a95e621febe7d1acb402b6068f975a&pb=U2FsdGVkX19V-bWbgCAsy4qge7CA2oQF6lBXPRtQz-MNZMZkdtcatn1jMBpFi1o_th5_ltM41rhn_qRsTsB_Kcy6rHTqxIQjj_CLNuwylEM')
data3 = req3.content
df_checks = pd.read_csv(BytesIO(data3), sep = ';')


# In[48]:


df_checks.head(5)


# In[15]:


# в датафрейме имеются данные о 541 транзакции
df_checks.info()


# In[55]:


# на каждого юзера приходится одна оплата
df_checks.student_id.nunique()


# In[16]:


df_checks.rev.hist();


# In[54]:


df_checks.rev.describe()


# В данном сете имеются данные о 541 транзакции 541 юзера. Предполагает ли система оплаты разовые платежи? Или наши данные представляют собой итоговые суммы платежей каждого юзера в исследуемый период? Не ясно, поэтому на будущее **метрики, учитывающие повторные платежи, мы использовать не будем**.  
# Само распределение чеков скошено влево, минимальный чек составил 199, а максимальный - 4650. Половина пользователей платит больше 840.

# ### Анализ пользовательской активности и оплат ###

# In[11]:


# получим общий датафрейм по активностям и оплатам
df_active_checks = df_active_studs.merge(df_checks, how = 'outer')


# In[18]:


# обратим внимание, что итоговый датасет содержит 8490 строк, что больше, чем 8431 (число активных юзеров)
# можно предположить, что множество плативших клиентов не полностью пересекается с множеством активных
# значит, есть те, кто платил, но не активничал
df_active_checks.info()


# In[19]:


# найдем юзеров, не вошедших в перечень активных, и при этом заплативших в дни эксперимента
df_active_checks[df_active_checks.activity.isna() & df_active_checks.rev>0]


# Есть 149 человек, которые **не были активны в дни эксперимента, но, тем не менее, оплатили приложение**. Посмотрим на разницу между группами активности, вдруг чего найдем.  
# Для удобства заменим NA в активности на 'no', а NA в чеках на 0

# In[12]:


df_active_checks.activity = df_active_checks.activity.fillna('no')
df_active_checks.rev = df_active_checks.rev.fillna(0)


# In[13]:


# посмотрим на размер чеков платящих юзеров среди активных и неактивных
sns.boxplot(x="activity", y="rev", data=df_active_checks.query('rev > 0'));


# In[14]:


# величина средних чеков платящих активных и неактивных юзеров
df_active_checks.query('rev > 0') \
.groupby('activity', as_index = False) \
.agg({'rev' : 'mean'})


# Активные клиенты чаще просто не платили (из 8431 юзера оплаты прошли только у 392), однако, при сравнении активных и неактивных платящих клиентов на первый взгляд кажется, что первые платят больше - распределение их платежей сдвинуто вправо (половина активных платит больше 990, в то время как по неактивным медиана составила 690, также различны максимальные платежи и средние).

# In[15]:


# посмотрим, как распределены тестовые группы внутри активных и неактивных юзеров
df_active_checks.merge(df_groups_total, how = 'left', on = 'student_id') \
.groupby(['activity', 'grp'], as_index = False) \
.agg({'student_id' : 'nunique'})


# Доли теста и контроля среди активных и неактивных, но платящих юзеров, примерно пропорциональны (в обеих группах порядка 80 процентов юзеров из тестовой группы).  
# Тогда интересно, насколько статзначимо различаются средние чеки платящих пользователей. Потому что если различаются, то можно осторожно предположить, что это не техническая ошибка учета юзеров, а скорее продуктовая особенность, и неактивные оплаты - что-то вроде ранее оформленных подписок (в случае ошибки регистрации активности пользователей их поведение в среднем не должно было бы отличаться).Поэтому возьмем распределения чеков в обеих группах и:  
# 1. **проверим на нормальность**,  
# 2. если они нормальны, **применим t-test**,  
# 3. иначе **применим бутстрап** для оценки среднего чека (критерий **Манна - Уитни рассматривать даже не будем**, так как наши группы диспропорциональны, а Манн - Уитни такое не жалует),   
# 4. **гипотеза Н0 - различий между группами нет, альтернативная гипотеза - они есть**, и средние чеки значимо отличаются в зависимости от активности юзеров.

# In[16]:


# проверка на нормальность чеков активных платящих пользователей
pg.normality(df_active_checks[(df_active_checks.activity == 'yes')&(df_active_checks.rev > 0)], method='normaltest')


# In[17]:


# проверка на нормальность чеков неактивных платящих пользователей
pg.normality(df_active_checks[(df_active_checks.activity == 'no')&(df_active_checks.rev > 0)], method='normaltest')


# Поскольку тест на нормальность показал p-value много меньше 0,05, считаем, что *оба распределения чеков значимо отличаются от нормального*. Применим бутстрап для оценки разницы ARPPU активных и неактивных пользователей:  
# **- Н0: средние чеки платящих активных и неактивных пользователей стат значимо не различаются,**  
# **- альтернативная гипотеза: между средними чеками платящих пользователей есть стат значимая разница**

# In[18]:


# Объявим функцию, которая позволит проверять гипотезу о равенстве метрик
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
            len(data_column_2), 
            replace = True # параметр возвращения
        ).values
        
        samples_2 = data_column_2.sample(
            len(data_column_2), 
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


# In[19]:


get_bootstrap(
    df_active_checks[(df_active_checks.activity == 'yes')&(df_active_checks.rev > 0)].rev, # числовые значения первой выборки
    df_active_checks[(df_active_checks.activity == 'no')&(df_active_checks.rev > 0)].rev, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
)


# p-value много меньше 0,05, следовательно имеется достаточно оснований отвергнуть основную гипотезу и принять альтернативную: *ARPPU активных и неактивных юзеров статзначимо различаются*. Предположим, что неактивные платящие пользователи - это ранее оформившие подписку и не заходившие в дни эксперимента на платформу.

# Отсюда далее **имеет смысл рассматривать только активных юзеров**, поскольку мы хотим оценить метрики с точки зрения воздействия новой системы оплаты, и пользователи, которые даже не заходили на платформу в дни эксперимента, ясности картине не добавят.  
# В любом случае, **для корректности оценки результатов А/Б теста необходимо разобраться с причиной данной ситуации**, поскольку из анализа вылетает почти треть платящих пользователей.

# ### Анализ тестовых групп ###

# In[23]:


# удалим неактивных юзеров с оплатами
df_active_checks['to_del'] = np.where((df_active_checks.activity == "no") & (df_active_checks.rev > 0), 'del', 'no del')
df_active_checks = df_active_checks.query('to_del == "no del"')
df_active_checks.drop(columns = 'to_del', axis = 1, inplace = True)


# In[24]:


# получим общий фрейм с экспериментальными группами, активностями и чеками
df = df_groups_total.merge(df_active_checks, how = 'outer', on = 'student_id')


# In[25]:


df.info()


# In[26]:


df.head()


# In[27]:


# для удобства заменим NA
df.activity = df.activity.fillna('no')
df.rev = df.rev.fillna(0)


# In[32]:


df.head()


# В случае, когда из 75 тысяч юзеров активными были 8431, и из них только 392 сделали покупку, ожидаемые распределения оплат внутри групп будут сильно скошены влево, с медианой около 0:

# In[30]:


df.rev[df.grp == 'A'].hist();


# In[31]:


df.rev[df.grp == 'A'].describe()


# In[34]:


df.rev[df.grp == 'B'].hist();


# In[48]:


df.rev[df.grp == 'B'].describe()


# In[38]:


# среди активных юзеров тестовой группы имеются бОльшие чеки
sns.boxplot(x="grp", y="rev", data=df[df.activity == "yes"]);


# In[40]:


# в среднем платящий юзер теста имеет бОльший чек относительно контроля
sns.boxplot(x="grp", y="rev", data=df.query('rev > 0'));


# ### Расчет метрик ###

# Ввиду того, что нам важно, изменила ли новая система оплаты поведение пользователей, стали ли они покупать чаще, или может быть больше, имеет смысл рассматривать следующие метрики:    
# - **CR в покупку активных пользователей** (частота покупок),  
# - **ARPAU** (средний чек активного пользователя),  
# - **ARPPU** (средний чек платящего пользователя).  
# ARPU считать не вижу смысла - оплат мало, пользователей много, метрика "размажется" и будет малочувствительной. Впрочем, конверсия скорее всего тоже не особо изменится по тем же причинам, но ее мы посчитаем как количественную метрику поведения юзеров. 

# In[ ]:


# добавим колонку для сравнения CR (есть покупка - значение 1, иначе - 0)
df['CR'] = df.rev.apply(lambda x: 1 if x > 0 else 0)


# In[46]:


# посчитаем указанные метрики:

# общее число юзеров
users_total = df.groupby('grp', as_index = False) \
.agg({'student_id' : 'nunique'}) \
.rename(columns = {'student_id' : 'users_total'})

# число платящих юзеров
users_rev = df.query('rev > 0') \
.groupby('grp', as_index = False) \
.agg({'student_id' : 'nunique'}) \
.rename(columns = {'student_id' : 'users_with_revenue'})

# число активных юзеров
users_active = df.query('activity == "yes"') \
.groupby('grp', as_index = False) \
.agg({'student_id' : 'nunique'}) \
.rename(columns = {'student_id' : 'users_active'})

# общая сумма платежей всех юзеров
rev_total = df.groupby('grp', as_index = False) \
.agg({'rev' : 'sum'}) \
.rename(columns = {'rev' : 'rev_total'})

# cумма платежей всех активных пользователей
users_active_rev = df.query('activity == "yes"') \
.groupby('grp', as_index = False) \
.agg({'rev' : 'sum'}) \
.rename(columns = {'rev' : 'users_active_rev'})

metrics = users_total.merge(rev_total, how = 'inner', on = 'grp') \
.merge(users_rev, how = 'inner', on = 'grp') \
.merge(users_active, how = 'inner', on = 'grp') \
.merge(users_active_rev, how = 'inner', on = 'grp')

metrics['CR'] = metrics.users_with_revenue / metrics.users_total
metrics['ARPAU'] = metrics.users_active_rev / metrics.users_active
metrics['ARPPU'] = metrics.rev_total / metrics.users_with_revenue

metrics


# Кажется, что средние чеки активных и платящих пользователей действительно выросли, а конверсия практически не изменилась

# ### Проверка гипотез о стат значимом росте метрик ###

# **Инструменты**:  
#     - t-test (требует нормального распределения и равномерности дисперсий),  
#     - Манн - Уитни (не любит диспропорции в сравниваемых выборках),  
#     - бутстрап.  
# Поскольку Манн - Уитни не подходит из-за различных размеров выборок, проверим распределение признаков на нормальность:

# **Гипотезы**:  
# - Н0: признак распределен нормально  
# - Н1: распределение признака значимо отличается от нормального

# In[47]:


# проверка на нормальность конверсии контрольной группы
pg.normality(df[df.grp == 'A'].CR, method='normaltest')


# In[51]:


# проверка на нормальность конверсии тестовой группы
pg.normality(df[df.grp == 'B'].CR, method='normaltest')


# In[48]:


# проверка на нормальность ARPAU контрольной группы
pg.normality(df[(df.grp == 'A')&(df.activity == "yes")].rev, method='normaltest')


# In[52]:


# проверка на нормальность ARPAU тестовой группы
pg.normality(df[(df.grp == 'B')&(df.activity == "yes")].rev, method='normaltest')


# In[49]:


# проверка на нормальность ARPPU контрольной группы
pg.normality(df[(df.grp == 'A')&(df.rev > 0)].rev, method='normaltest')


# In[53]:


# проверка на нормальность ARPPU тестовой группы
pg.normality(df[(df.grp == 'B')&(df.rev > 0)].rev, method='normaltest')


# Во всех случаях значения p-value меньше 0,05, гипотеза о нормальном распределении признаков во всех случаях отвергается: **конверсия и оплаты активных и платящих пользователей ненормально распределены**  
# Поэтому применим бутстрап:

# In[54]:


# посмотрим на бутстрэп по CR в покупку среди активных пользователей
# Н0: конверсии в тестовой и контрольной группе значимо не различаются
# Н1: конверсии теста и контроля значимо отличны
get_bootstrap(
    df[df.grp == 'A'].CR, # числовые значения первой выборки
    df[df.grp == 'B'].CR, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
)


# pvalue = 0.85, что не позволяет отвергнуть основную гипотезу: **между тестом и контролем по конверсии в покупку статистически значимых различий не наблюдается**

# In[57]:


# посмотрим на бутстрэп по ARPAU
# Н0: средний чек активного юзера в тестовой и контрольной группе значимо не различаются
# Н1: средний чек активного юзера теста и контроля значимо отличны
get_bootstrap(
    df[(df.grp == 'A')&(df.activity == "yes")].rev, # числовые значения первой выборки
    df[(df.grp == 'B')&(df.activity == "yes")].rev, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
)


# p-value < 0.05, что позволяет отвергнуть основную гипотезу об отсутствии значимых различий, **активные пользователи тестовой группы платят больше.**

# In[60]:


# посмотрим на бутстрэп по ARPPU
# Н0: средний чек платящего юзера в тестовой и контрольной группе значимо не различаются
# Н1: средний чек платящего юзера теста и контроля значимо отличны
get_bootstrap(
    df[(df.grp == 'A')&(df.rev > 0)].rev, # числовые значения первой выборки
    df[(df.grp == 'B')&(df.rev > 0)].rev, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
)


# p-value < 0.05, что позволяет отвергнуть основную гипотезу об отсутствии значимых различий, **у платящих пользователей тестовой группы чек в среднем больше.**

# ### Выводы: ###
# 1. Необходимо прояснить ситуацию вокруг 149 юзеров с неактивной оплатой,  
# 2. Требуется дополнительный анализ системы сплитования юзеров на тест и контроль, а так же уточнение, нет ли путаницы в описании датасета,  
# 3. Если входные данные верны, то новая система оплаты не побуждает пользователя платить чаще,  
# 4. С тем же допущением верно то, что если пользователь платит - он платит статистически значимо больше! мы не видим новой системы оплаты, но возможно, что она связана с бонусами или перекрестной лояльностью к партнерам например, или какие-то всплывающие скидки из категории "только сегодня и только сейчас / гипс снимают, клиент уезжает", побуждающие покупать больше. И это статистически значимо.  
# После уточнения пп. 1 и 2 новая система может быть расширена на всех пользователей.

# ## Задание 2. SQL ##

# Образовательные курсы состоят из различных уроков, каждый из которых состоит из нескольких маленьких заданий. Каждое такое маленькое задание называется "горошиной". Назовём очень усердным учеником того пользователя, который хотя бы раз за текущий месяц правильно решил 20 горошин.
# Необходимо написать **оптимальный запрос, который даст информацию о количестве очень усердных студентов за март 2020 года.**

# Дана таблица default.peas:  
# - st_id(int) - ID ученика,  
# - timest(timestamp) - Время решения карточки,  
# - correct(bool) - Правильно ли решена горошина?,  
# - subject(text) - Дисциплина, в которой находится горошина

# In[1]:


import pandahouse as ph


# In[2]:


#объявляем параметры подключения
connection = dict(database='default',
                  host='https://clickhouse.lab.karpov.courses',
                  user='student',
                  password='dpo_python_2020')


# Обратим внимание, что нам доступны данные только за 2 дня 30.10.2021 и 31.10.2021, за март 2020 года усердных учеников мы просто не получим.Однако, сформируем вопрос в соответствии с заданием:

# In[89]:


#пишем запрос, и получаем данные из clickhouse в pandas dataframe
query = """
SELECT
    st_id,
    SUM(correct) AS sum_pease
FROM (
    SELECT
        st_id,
        toStartOfMonth(timest) AS MonthBegin,
        toInt32(correct) AS correct
    FROM default.peas
    WHERE MonthBegin == toDate('2020-03-01')   
)
GROUP BY st_id
HAVING sum_pease >=20
"""
df = ph.read_clickhouse(query, connection=connection)
df


#  ## Второе задание SQL ##
# Необходимо в одном запросе выгрузить следующую информацию о группах пользователей:  
# - ARPU (по всем пользователям) 
# - ARPAU (по пользователям, решившим больше 10 задач правильно в любых дисциплинах) 
# - CR в покупку (по всем пользователям) 
# - СR активного пользователя в покупку (юзеры с более 10 задач верных) 
# - CR пользователя из активности по математике (subject = ’Math’) в покупку курса по математике (юзеры с решенными верно 2 или больше задач по математике)

# **Подзапросы:**  
# - Для ARPU и CR нам нужна табличка с айди юзеров, суммами их покупок и тестовыми группами  
# - Для ARPAU CR активных нужно выбрать айдишники юзеров, решивших больше 10 задач, и также сджойнить таблицы с покупками и тестовыми группами  
# - Для CR математики нужно выбрать айди юзеров, решивших больше 2 задач по математике, и приджойнить к ним покупки математики  
# 
# Итог: все айдишники юзера, тестовая группа, сумма покупки, сумма покупки по математике, число решенных задач, число решенных задач по математике

# In[6]:


#пишем запрос, и получаем данные из clickhouse в pandas dataframe
query = """
WITH total_money_id AS (SELECT st_id, SUM(money) AS total_money FROM default.final_project_check GROUP BY st_id),
    total_money_math_id AS (SELECT st_id, SUM(money) AS total_money_math FROM default.final_project_check GROUP BY st_id HAVING subject == 'Math'),
    total_peas_id AS (SELECT st_id, SUM(correct) AS total_peas FROM default.peas GROUP BY st_id),
    total_peas_math_id AS (SELECT st_id, SUM(correct) AS total_peas_math FROM default.peas GROUP BY st_id HAVING subject == 'Math')

SELECT
    test_grp,
    SUM(total_money) / COUNT(st_id) AS ARPU,
    SumIf(total_money, total_peas > 10) / countIf(st_id, total_peas > 10) AS ARPAU,
    countIf(total_money>0) / COUNT(DISTINCT st_id) AS CR,
    countIf(st_id, total_money_math>0 AND total_peas > 10) / countIf(st_id, total_peas > 10) AS CR_active,
    countIf(st_id, total_money_math>0 AND total_peas_math >=2) / countIf(DISTINCT st_id, total_peas_math >=2) AS CR_math
FROM
    (SELECT
                    a.st_id AS st_id,
                    a.test_grp AS test_grp,
                    a.total_money AS total_money,
                    a.total_money_math AS total_money_math,
                    a.total_peas AS total_peas,
                    b.total_peas_math AS total_peas_math
    FROM
        (SELECT
                    a.st_id AS st_id,
                    a.test_grp AS test_grp,
                    a.total_money AS total_money,
                    a.total_money_math AS total_money_math,
                    b.total_peas AS total_peas
        FROM    
            (SELECT
                    a.st_id AS st_id,
                    a.test_grp AS test_grp,
                    a.total_money AS total_money,
                    b.total_money_math AS total_money_math
            FROM (SELECT
                    a.st_id AS st_id,
                    a.test_grp AS test_grp,
                    b.total_money AS total_money
                FROM default.studs AS a
                LEFT JOIN total_money_id as b
                ON a.st_id == b.st_id) AS a
            LEFT JOIN total_money_math_id AS b
            ON a.st_id == b.st_id) AS a
        LEFT JOIN total_peas_id AS b
        ON a.st_id == b.st_id) AS a
    LEFT JOIN total_peas_math_id AS b
    ON a.st_id == b.st_id)
GROUP BY test_grp
"""
df = ph.read_clickhouse(query, connection=connection)
df


# ## Задание 3. Функции ##

# Реализовать функции, которые будут:  
# - автоматически подгружать информацию из дополнительного файла groups_add.csv (заголовки могут отличаться)  
# - на основании дополнительных параметров пересчитывать метрики  
# - строить графики по получаемым метрикам.  
# 
# Предположения исходя их формулировки задачи: 
# 0. В связи с необходимостью автоматизации я написала даг, который собирает данные, мерджит и возвращает таблицу с метриками. При необходимости таски дага можно переделать в отдельные функции, но тогда сбор будет не автоматический.
# 1. Дополнительно подгружаемый файл аналогичен 1 заданию и содержит только айдишки добавляемых пользователей и их группы
# 2. Все файлы лежат по ранее указанным путям
# 3. Остальная информация - об активностях и оплатах - лежит в уже исследованных файлах df_active_studs и df_checks  

# In[67]:


default_args = {
    'owner': 'e.malofeeva',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2022, 7, 31),
    'schedule_interval' : '0 24 * * *'
}


@dag(default_args = default_args, catchup = False)
def update_metrics():
    # таска считывания groups
    @task()
    def get_data_groups():
        req1 = requests.get('https://s856sas.storage.yandex.net/rdisk/68e81c82e181de5daf7388627f052597b13b536aa0a80fef5ee9ed42c29fd847/62ffbd8d/fKqInKw3d7bLFOeFnMGnhLWIS9AmXMMXWGR17A-BkO3-qg5Wp8tUAp7lxoJJvJyG8pjltSuuxShICmGl2FDiJhbsM4PC04KdB-NoZUjfeOOr8npumZHI4midPdWhecNq?uid=0&filename=%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82_2_groups.csv&disposition=attachment&hash=FjLs%2B6AJdyaRoyFMFeL4raj9RoBP3hYIuXfguWvuEGcjdjYEIrmN7ZN/eZ32Z77Cq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=text%2Fplain&owner_uid=125175473&fsize=724984&hid=4b9098e1d545e562c779de19f9f64e2f&media_type=spreadsheet&tknv=v2&rtoken=hnecBu3cpsUk&force_default=no&ycrid=na-2b0f5c1337be7f69292be597866703f9-downloader21h&ts=5e69aca10bd40&s=0494cd844e1de7e7cd86ef2596ed29eb129ca6f9be7725452a0c695808e7636a&pb=U2FsdGVkX196IfjqXhBcL2stXy2LnkQ7NVKd0kJLP_0-SOAlIGqhpZytKXChCTr_taw1cyV4-HCoJZq2x7i-IM6Cn1-mjfxFb9VT1BKqfptmlay7nPbkfLszesK5TypY')
        data1 = req1.content
        df_groups = pd.read_csv(BytesIO(data1), header = None, skiprows = 1, names = ['student_id', 'grp'], sep = ';')
        return df_groups
    
        # таска считывания active_studs
    @task()
    def get_data_active_studs():
        req2 = requests.get('https://s432vla.storage.yandex.net/rdisk/3a40f377f696432a2588a986b2af49044dd68afb5b18d69010f8282c1f40f146/62ffbec8/n3N0SQYM2SoDHEJIQgKt2ujEDw5jYN17sSz14UE4iwGyYlrHRmnPoV9WI8mqa1dmF6Qt_88ffG90wBmmEpDhFQ==?uid=0&filename=%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82_2_active_studs.csv&disposition=attachment&hash=zWS0uLxK3MRXd/4/IErmDFQBGvbpGD1qm/aJx/k6EUUtE6DaDI0uGkDT4cHh/WJbq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=text%2Fplain&owner_uid=125175473&fsize=65131&hid=3ed23002f82477fd8c7d4a21c17d8034&media_type=spreadsheet&tknv=v2&rtoken=VLQnNMBdZHav&force_default=no&ycrid=na-96300cdc146eb8e9f9841f7d1c03ae4f-downloader11f&ts=5e69adcd74200&s=298f3f9a6a0b4dcfcd30b5e2531375074e1817b4485c4e2acc2cdda94c064471&pb=U2FsdGVkX19nAru0O76sUeX5I7sjU7s3OfVwfBSgj3bVELERwhe44uxz5-Qbg-huJf5HKtETRz_t9FsHXfXWWfoU1L7utUUpsCWTRvpOBWo')
        data2 = req2.content
        df_active_studs = pd.read_csv(BytesIO(data2), header = None, skiprows = 1, names = ['student_id'])
        df_active_studs['activity'] = 'yes'
        return df_active_studs
    
        # таска считывания checks
    @task()
    def get_data_checks():
        req3 = requests.get('https://s251vla.storage.yandex.net/rdisk/2ed0cef7d3eeb388a1cfef16c898d547abd95ff3590fc483208d4d0329f39e40/62ffbf91/n3N0SQYM2SoDHEJIQgKt2psRrG10OKyAasqgpda4iX7NDTSjoP6nmHZtzOqJXgQH436kYOx6cWVM8zgvGq_zHg==?uid=0&filename=%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82_2_checks.csv&disposition=attachment&hash=TeGedpkC9yRvIkaaNDf1zC3bFkr2Kvo9MvW7kYjfHc5svktv83UZ47kn6%2BaauC1pq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=text%2Fplain&owner_uid=125175473&fsize=7764&hid=caf94adff3374e1c1a09fba3aae11e50&media_type=spreadsheet&tknv=v2&rtoken=lCY7FQdHOPMD&force_default=no&ycrid=na-5f259d0cfa6acec9fbf8a1703b68f794-downloader2e&ts=5e69ae8d24640&s=5e3efcf73091f37afe750f8ddb7beb1773a95e621febe7d1acb402b6068f975a&pb=U2FsdGVkX19V-bWbgCAsy4qge7CA2oQF6lBXPRtQz-MNZMZkdtcatn1jMBpFi1o_th5_ltM41rhn_qRsTsB_Kcy6rHTqxIQjj_CLNuwylEM')
        data3 = req3.content
        df_checks = pd.read_csv(BytesIO(data3), header = None, skiprows = 1, names = ['student_id', 'rev'], sep = ';')
        return df_checks
    
    # таска считывания add
    @task()
    def get_data_groups_add():
        req4 = requests.get('https://s750sas.storage.yandex.net/rdisk/f06bf3fb0ce3d7ed783c715aa18a0389af6e9d4efa00d64ca5f53aac6182abf1/62ffc01e/fKqInKw3d7bLFOeFnMGnhGAdJ2MSdW82i4_ghmwgsZlWhJkpRytn07Ls4kX484X-lezQy4loqSz9AFrIeHDnv8QEHwKSTiYbb_hzq1q6XQir8npumZHI4midPdWhecNq?uid=0&filename=%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82_2_group_add.csv&disposition=attachment&hash=0ertv33lnyTdzTu9NbdN9MtgpuTSXleSHZ%2BvhK85v05jGwJs7DHSrj4ununKsRcfq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=text%2Fplain&owner_uid=125175473&fsize=949&hid=e8dcec35987c367a56f82404a22463b7&media_type=spreadsheet&tknv=v2&rtoken=x7sQvDBvEcVd&force_default=no&ycrid=na-6ae8cf4f9feb8da91b76dd02295986e5-downloader2e&ts=5e69af139c380&s=28a8f791e2b55bce136323bbd03a635c3fae77f3b10cbcaff7b7a06d71e43fd6&pb=U2FsdGVkX18Wzaw8soN5pjC05gU8iWY8NCgpwE3Bgchzktb4Mdsjxki6tJpzFvDFmTxaIG_MODh8nBJa5LDlTyyWkkVUpW84A0_EDWJ33f5Fytt1LWGLsoVjXXtP2sIv')
        data4 = req4.content
        df_groups_add = pd.read_csv(BytesIO(data4), header = None, skiprows = 1, names = ['student_id', 'grp'])
        return df_groups_add

    # таска преобразования фреймов
    @task()
    def union_df(df_groups, df_groups_add, df_checks, df_active_studs):
        df_groups = pd.concat([df_groups, df_groups_add])
        df = df_groups.merge(df_active_studs, how = 'left', on = 'student_id').fillna('no')
        df = df.merge(df_checks, how = 'left', on = 'student_id').fillna(0)
        return df

    # таска расчета метрик
    @task()
    def metrics(df):
       # общее число юзеров
        users_total = df.groupby('grp', as_index = False) \
        .agg({'student_id' : 'nunique'}) \
        .rename(columns = {'student_id' : 'users_total'})

        # число платящих юзеров
        users_rev = df.query('rev > 0') \
        .groupby('grp', as_index = False) \
        .agg({'student_id' : 'nunique'}) \
        .rename(columns = {'student_id' : 'users_with_revenue'})

        # число активных юзеров
        users_active = df.query('activity == "yes"') \
        .groupby('grp', as_index = False) \
        .agg({'student_id' : 'nunique'}) \
        .rename(columns = {'student_id' : 'users_active'})

        # общая сумма платежей всех юзеров
        rev_total = df.groupby('grp', as_index = False) \
        .agg({'rev' : 'sum'}) \
        .rename(columns = {'rev' : 'rev_total'})

        # cумма платежей всех активных пользователей
        users_active_rev = df.query('activity == "yes"') \
        .groupby('grp', as_index = False) \
        .agg({'rev' : 'sum'}) \
        .rename(columns = {'rev' : 'users_active_rev'})

        metrics = users_total.merge(rev_total, how = 'inner', on = 'grp') \
        .merge(users_rev, how = 'inner', on = 'grp') \
        .merge(users_active, how = 'inner', on = 'grp') \
        .merge(users_active_rev, how = 'inner', on = 'grp')

        metrics['CR'] = metrics.users_with_revenue / metrics.users_total
        metrics['ARPAU'] = metrics.users_active_rev / metrics.users_active
        metrics['ARPPU'] = metrics.rev_total / metrics.users_with_revenue

        return metrics.to_csv('metrics.csv')
        
    df_groups = get_data_groups()
    df_active_studs = get_data_active_studs()
    df_checks = get_data_checks()
    df_groups_add = get_data_groups_add()
    df = union_df(df_groups, df_groups_add, df_checks, df_active_studs)
    metrics = metrics(df)
    
update_metrics = update_metrics()


# In[102]:


# Функция рисования графиков по полученным метрикам:
def metrics_graph(df):
    sns.set(rc={'figure.figsize':(20,12)})
    fig, axes = plt.subplots(nrows=1, ncols=3)
    sns.barplot(x=df.grp, y=df.CR, ax = axes[0], data=df)
    sns.barplot(x=df.grp, y=df.ARPAU, ax = axes[1], data=df)
    sns.barplot(x=df.grp, y=df.ARPPU, ax = axes[2], data=df)


# In[103]:


metrics_graph(metrics)

