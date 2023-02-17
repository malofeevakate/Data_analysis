#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from operator import attrgetter
import matplotlib.colors as mcolors
from datetime import timedelta


# ### Посмотрим на таблицу с уникальными идентификаторами пользователей

# In[2]:


# customer_id — позаказный идентификатор пользователя
# customer_unique_id —  уникальный идентификатор пользователя  (аналог номера паспорта)
# customer_zip_code_prefix —  почтовый индекс пользователя
# customer_city —  город доставки пользователя
# customer_state —  штат доставки пользователя
df_customers = pd.read_csv('olist_customers_dataset.csv')


# In[3]:


df_customers.head()


# In[4]:


df_customers.shape


# In[5]:


# пропущенные значения отсутствуют
df_customers.isna().sum()


# In[6]:


# сравним число юников айди юзеров и айди юзеров, связанных с заказами
df_customers.customer_id.nunique()


# In[7]:


df_customers.customer_unique_id.nunique()


# Мы видим, что юников меньше, чем айди юзеров, связанных с заказами, что логично (некоторые пользователи делают больше одного заказа)

# ## Посмотрим на таблицу с данными заказов

# In[8]:


# order_id —  уникальный идентификатор заказа (номер чека)
# customer_id —  позаказный идентификатор пользователя
# order_status —  статус заказа
# order_purchase_timestamp —  время создания заказа
# order_approved_at —  время подтверждения оплаты заказа
# order_delivered_carrier_date —  время передачи заказа в логистическую службу
# order_delivered_customer_date —  время доставки заказа
# order_estimated_delivery_date —  обещанная дата доставки
df_orders = pd.read_csv('olist_orders_dataset.csv')


# In[9]:


df_orders.head()


# In[10]:


df_orders.shape


# In[11]:


# посмотрим на типы данных, в первую очередь нас интересует тип временных переменных
df_orders.dtypes


# In[12]:


# преобразуем форматы временных переменных в timestamp
df_orders.order_purchase_timestamp = pd.to_datetime(df_orders.order_purchase_timestamp)
df_orders.order_approved_at = pd.to_datetime(df_orders.order_approved_at)
df_orders.order_delivered_carrier_date = pd.to_datetime(df_orders.order_delivered_carrier_date)
df_orders.order_delivered_customer_date = pd.to_datetime(df_orders.order_delivered_customer_date)
df_orders.order_estimated_delivery_date = pd.to_datetime(df_orders.order_estimated_delivery_date)


# In[13]:


df_orders.isna().sum()


# Обратим внимание, что имеются пропущенные данные в датах перемещения заказов:  
#     - из всех созданных заказов по 160 заказам не подтвердилась оплата,  
#     - 1783 заказа не переданы в доставку,  
#     - 2965 заказов не доставлены конечным пользователям

# In[14]:


# а что вобще по статусам заказов?
df_orders.order_status.value_counts()


# In[15]:


# посмотрим подробнее на заказы со статусом delivered:
df_orders.query('order_status == "delivered"').isna().sum()


# In[16]:


df_orders.query('order_delivered_customer_date == order_delivered_customer_date').value_counts('order_status')


# In[17]:


df_orders.groupby('order_status', as_index = False).count()


# Анализ данных в разрезе статусов заказов показал, что порядка 97 % заказов с датой доставки клиенту имеют статус delivered. При этом, если опираться только на статус delivered, мы видим, что:  
# - доставлено 14 заказов с отсутствующей датой подтверждения оплаты. Все они приходятся на даты заказов 19.01.2017, 17.02.2017 - 19.02.2017, разные юники, разные города. Если предположить, что эти юзеры - должники нашей системы (товар получили, но не оплатили), значит, эта опция доступна какому-то количеству пользователей, почему воспользовавшихся ею так мало за рассматриваемый период? Скорее всего какой - то баг в системе подтверждения оплаты или ее учета, поэтому оставляю в выборке доставленных и оплаченных заказов), 
# - 2 заказа с отсутствующей датой передачи заказа в логистику,  
# - 8 заказов с отсутствующей датой вручения клиенту (почти все невручения приходятся на один и тот же штат - Сан Паулу, вроде и небольшая доля в общем объеме наших заказов, но брать их в выборку купленных не хочется).
# Ввиду того, что покупка - это оплата и получение товара клиентом (в совокупности), а также необходимости сделать метрику более чувствительной к отказам от покупки заказов на разных этапах, целесообразно **признать покупкой транзакцию, у которой статус "доставлено" + имеется дата передачи клиенту**

# ## Посмотрим на таблицу с данными товарных позиций, входящих в заказы

# In[18]:


# order_id —  уникальный идентификатор заказа (номер чека)
# order_item_id —  идентификатор товара внутри одного заказа
# product_id —  ид товара (аналог штрихкода)
# seller_id — ид производителя товара
# shipping_limit_date —  максимальная дата доставки продавцом для передачи заказа партнеру по логистике
# price —  цена за единицу товара
# freight_value —  вес товара
df_items = pd.read_csv('olist_order_items_dataset.csv')


# In[19]:


df_items.head()


# In[20]:


df_items.shape


# In[21]:


df_items.dtypes


# In[22]:


# переведем планируемую дату доставки в более подходящий формат
df_items.shipping_limit_date = pd.to_datetime(df_items.shipping_limit_date)


# In[23]:


df_items.isna().sum()


# что такое order_item_id? посмотрим на примере order_id == 00143d0f86d6fbd9f9b38ab440ac16f5:

# In[24]:


# данный заказ хранится в таблице заказов
df_orders.query('order_id == "00143d0f86d6fbd9f9b38ab440ac16f5"')


# In[25]:


# и в таблице товаров, входящих в заказы. При чем мы видим, что в данный заказ входит 3 единицы одного и того же товара
# таким образом, в order_item_id хранится порядковый номер заказанного товара (а количество строк в данном заказе = количество товаров)
df_items.query('order_id == "00143d0f86d6fbd9f9b38ab440ac16f5"')


# Заметим, что в таблице товаров, входящих в заказы, есть информация не обо всех заказах (по 775 заказам данных нет):

# In[26]:


df_orders.order_id.nunique()


# In[27]:


df_items.order_id.nunique()


# ### Для определения пользователей, совершивших покупку только один раз, смержим дату клиентов с датой заказов:

# In[28]:


df_customers_orders = df_orders.merge(df_customers)


# In[29]:


df_customers_orders.head()


# In[30]:


df_customers_orders.shape


# In[31]:


df_customers_orders.query('(order_status == "delivered") & (order_delivered_customer_date == order_delivered_customer_date)') \
.groupby('customer_unique_id', as_index = False) \
.agg({'customer_id' : 'count'}) \
.rename(columns = {'customer_id' : 'count_orders'}) \
.query('count_orders < 2')


# In[32]:


df_customers_orders['year'] = df_customers_orders.order_purchase_timestamp.dt.year


# In[33]:


df_customers_orders.query('(order_status == "delivered") & (order_delivered_customer_date == order_delivered_customer_date)') \
.groupby('customer_unique_id', as_index = False) \
.agg({'customer_id' : 'count'}) \
.rename(columns = {'customer_id' : 'count_orders'}) \
.hist()


# In[34]:


df_customers_orders.query('(order_status == "delivered") & (order_delivered_customer_date == order_delivered_customer_date)') \
.groupby('customer_unique_id', as_index = False) \
.agg({'customer_id' : 'count'}) \
.rename(columns = {'customer_id' : 'count_orders'}) \
.query('count_orders < 2')


# Итак, больше 90 % наших юзеров (90 549) совершает покупку единожды

# ### Сколько заказов в месяц в среднем не доставляется по разным причинам (вывести детализацию по причинам):

# какие заказы мы можем считать недоставленными? Те, у кого нет статуса delivered, на дату доставки не смотрим, так как выше мы установили, что дата доставки имеется у двух видов заказов - доставленных и отмененных, а вторые нам потребуются

# In[35]:


df_not_delivered = df_orders.query('order_status != "delivered"') \
.groupby('order_status', as_index = False) \
.agg({'order_id' : 'count'}) \
.rename(columns={'order_id': 'total_not_delivered'})


# In[36]:


# а сколько всего месяцев в рассматриваемой выборке?
total_month= (df_orders.order_purchase_timestamp.max() - df_orders.order_purchase_timestamp.min())/np.timedelta64(1, 'M')


# In[37]:


df_not_delivered['avg_not_delivered_per_month'] = df_not_delivered.total_not_delivered / total_month


# In[38]:


df_not_delivered.sort_values('avg_not_delivered_per_month', ascending = False)


# Видно, что наибольшая доля недоставленных заказов застревает на этапе отгрузки со склада. Так же в топ - 3 есть отмененные заказы, и по каким-то причинам недоступные. Показательно, что доля неподтвержденных и неоплаченных заказов минимальна.

# ### Для опредения топового дня недели по каждому товару смерджим дату заказов с датой товаров в этих заказах (при этом учитываем, что по 775 заказам нет данных об их составе в таблице df_items):

# In[39]:


df_items_orders = df_items.merge(df_orders)


# In[40]:


df_items_orders.head()


# In[41]:


df_items_orders['weekday'] = df_items_orders.order_purchase_timestamp.dt.day_name()


# In[42]:


df_products_weekday = pd.pivot_table(df_items_orders, index = 'product_id', columns = 'weekday', values = 'order_id', aggfunc = 'count').fillna(0)


# In[43]:


df_products_weekday


# In[44]:


df_products_weekday['top_day'] = df_products_weekday.idxmax(axis=1)


# In[45]:


# результат представлен в колонке weekday. Следует обратить внимание, что в некоторых случаях топ-дней у того или иного товара более одного (например, товар с product_id = 000b8f95fcb9e0096488278317764d19)
pd.DataFrame(df_products_weekday.where(df_products_weekday.eq(df_products_weekday.max(1), axis=0)).stack()).rename(columns={0 : 'count_orders'})


# ### Сколько у каждого из пользователей в среднем покупок в неделю (по месяцам)? Не стоит забывать, что внутри месяца может быть не целое количество недель. Например, в ноябре 2021 года 4,28 недели. И внутри метрики это нужно учесть.

# Посчитаем общее количество покупок по каждому пользователю в месяц, и поделим эти суммы на число недель в соответствующем месяце

# In[46]:


# выберем нужные колонки (юник и дата покупки) для тех пользователей, которые, как мы ранее определили, совершили завершенные покупки
df_customers_order_purchase = df_customers_orders.query('(order_status == "delivered") & (order_delivered_customer_date == order_delivered_customer_date)')[['customer_unique_id', 'order_purchase_timestamp']]


# In[47]:


df_customers_order_purchase


# In[48]:


# создадим колонки с годом и месяцем
df_customers_order_purchase['month'] = df_customers_order_purchase.order_purchase_timestamp.dt.month
df_customers_order_purchase['year'] = df_customers_order_purchase.order_purchase_timestamp.dt.year


# In[49]:


df_customers_order_purchase.head()


# In[50]:


# сделаем справочную таблицу с количеством недель в соответствующих месяцев
df_m_y = pd.DataFrame(df_customers_order_purchase.order_purchase_timestamp)


# In[51]:


df_m_y


# In[52]:


df_m_y['month'] = df_m_y.order_purchase_timestamp.dt.month
df_m_y['year'] = df_m_y.order_purchase_timestamp.dt.year


# In[53]:


df_m_y.month = df_m_y.month.apply(lambda x: str(x))
df_m_y.year = df_m_y.year.apply(lambda x: str(x))


# In[54]:


df_m_y


# In[55]:


df_m_y['m_y'] = df_m_y.year + '/' + df_m_y.month


# In[56]:


df_m_y['day_in_month'] = df_m_y.order_purchase_timestamp.dt.daysinmonth


# In[57]:


df_m_y['weeks_in_month'] = round((df_m_y.day_in_month / 7), 2)


# In[58]:


df_m_y


# In[59]:


# результирующая справочная таблица
df_month_week = df_m_y.groupby('m_y', as_index = False) \
.agg({'weeks_in_month' : 'mean'})


# In[60]:


# получим количество покупок в месяц для каждого юника
df_customers_count_orders = df_customers_order_purchase.groupby(['customer_unique_id', 'month', 'year'], as_index = False) \
.agg({'order_purchase_timestamp' : 'count'}) \
.rename(columns = {'order_purchase_timestamp' : 'count_orders'}) \
.sort_values('count_orders', ascending = False)


# In[61]:


df_customers_count_orders


# In[62]:


df_customers_count_orders.month = df_customers_count_orders.month.apply(lambda x: str(x))
df_customers_count_orders.year = df_customers_count_orders.year.apply(lambda x: str(x))


# In[63]:


df_customers_count_orders['m_y'] = df_customers_count_orders.year + '/' + df_customers_count_orders.month


# In[64]:


# таблица по юникам для мерджа справочной таблицы с количеством недель
df_customers_count_orders


# In[65]:


# мерджим к таблице юников справочную инфу
df_orders_week = df_customers_count_orders.merge(df_month_week, how = 'left', on = 'm_y')


# In[66]:


df_orders_week.head()


# In[67]:


# создаем соответствующий столбец
df_orders_week['count_orders_week'] = df_orders_week.count_orders / df_orders_week.weeks_in_month


# In[68]:


# результирующие данные содержатся в столбце count_orders_week
df_orders_week.sort_values('customer_unique_id', ascending = False)


# In[69]:


df_orders_week.count_orders_week.hist()


# ### Когортный анализ пользователей за период с января по декабрь (найти  когорту с самым высоким retention на 3й месяц)

# Поскольку на предыдущем этапе мы заметили, что более 90 % наших клиентов сделали только по одной покупке, можно ожидать маленький ретеншн. Возьмем пользователей, совершивших покупки в 2017 году

# In[70]:


df_cohort = df_customers_orders.query('(order_status == "delivered") & (order_delivered_customer_date == order_delivered_customer_date)')


# In[71]:


# для фильтрации данных за 2017 год добавим графу год, и отберем по ней интересующих нас юников
df_cohort['year'] = df_cohort.order_purchase_timestamp.dt.year


# In[72]:


df_cohort = df_cohort.query('year == "2017"')


# In[73]:


df_cohort


# In[74]:


# сохраняем только релевантные столбцы - юники и дата заказа
df = df_cohort[['customer_unique_id', 'order_purchase_timestamp']]


# In[75]:


# создадим колонку с указанием когорты (месячная когорта на основе даты первой покупки)
df['cohort'] = df.groupby('customer_unique_id')['order_purchase_timestamp'] \
.transform('min') \
.dt.to_period('M')


# In[76]:


# добавим усеченный месяц даты покупки
df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M')


# In[77]:


df


# In[78]:


# посчитаем распределение юников по когортам
df1 = df.groupby(['cohort', 'order_month']) \
.agg({'customer_unique_id' : 'nunique'}) \
.rename(columns = {'customer_unique_id' : 'n_customers'}) \
.reset_index(drop = False)


# In[79]:


df1


# In[80]:


df1['period_number'] = (df1.order_month - df1.cohort).apply(attrgetter('n'))


# In[81]:


df1


# In[82]:


df_cohort_pivot = df1.pivot_table(index = 'cohort',
                                     columns = 'period_number',
                                     values = 'n_customers')


# In[83]:


df_cohort_pivot


# In[84]:


# расчитаем ретеншн в процентах, поскольку абсолютные значения крайне невелики
cohort_size = df_cohort_pivot.iloc[:,0]
retention_matrix = df_cohort_pivot.divide(cohort_size, axis = 0)*100


# In[85]:


retention_matrix


# In[86]:


with sns.axes_style("white"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True, gridspec_kw={'width_ratios': [1, 11]})
    
    # retention matrix
    sns.heatmap(retention_matrix, 
                mask=retention_matrix.isnull(), 
                annot=True, 
                fmt='.3g', 
                cmap="viridis", 
                ax=ax[1])
    ax[1].set_title('Monthly Cohorts: User Retention', fontsize=16)
    ax[1].set(xlabel='# of periods',
              ylabel='')

    # cohort size
    cohort_size_df = pd.DataFrame(cohort_size).rename(columns={0: 'cohort_size'})
    white_cmap = mcolors.ListedColormap(['white'])
    sns.heatmap(cohort_size_df, 
                annot=True, 
                cbar=False, 
                fmt='g', 
                cmap="viridis", 
                ax=ax[0])

    fig.tight_layout()


# Мы видим, что при вводной "покупка = дата доставки + статус", наибольший ретеншн за 3й месяц приходится на когорту 2017 - 06. Однако нельзя не заметить, что по всему периоду по всем когортам ретеншн крайне невелик.

# ### RFM-сегментация пользователей

# - recency - насколько недавно была совершена последняя покупка клиента (количество дней с момента последней покупки),  
# - frequency - как часто клиент совершает покупку (количество покупок за исследуемый период (обычно один год)),    
# - monetary - сколько денег тратит клиент (общая сумма покупок, сделанных за исследуемый период),  
# - найти квинтили для каждого из этих измерений,  
# - дайте оценку каждому параметру в зависимости от того, в каком квинтиле он находится,  
# - объединить баллы R, F и M, чтобы получить балл RFM,  
# - сопоставить оценки RF с сегментами

# In[112]:


# поскольку для расчета показателей RFM нам нужны как айдишники заказов и юников, так и цена, смержим все таблицы
df = df_items_orders.merge(df_customers)


# In[113]:


# общая дата по совершенным покупкам
df = df.query('(order_status == "delivered") & (order_delivered_customer_date == order_delivered_customer_date)')


# In[114]:


# посмотрим на общие суммы заказов в разрезе пользователей и дат
df = df.groupby(['customer_unique_id', 'order_id', 'order_purchase_timestamp'], as_index = False) \
.agg({'price' : 'sum'})


# In[115]:


print('Рассматриваемый период с {} по {}'.format(df1['order_purchase_timestamp'].min(),
                                    df1['order_purchase_timestamp'].max()))


# In[116]:


period = 365


# In[117]:


# так как данные старые, возьмем последнюю дату, от которой считается recency, как максимальную дату в полученном датасете + 1
last_date = df['order_purchase_timestamp'].max() + timedelta(days = 1)


# In[118]:


last_date


# In[119]:


# порежем датасет по полученной границе периода
df = df[df['order_purchase_timestamp'] >= last_date - timedelta(days = period)]


# In[120]:


# проверим
df.order_purchase_timestamp.min()


# In[121]:


df.order_purchase_timestamp.max()


# In[122]:


# посчитаем разницу между концом периода и датой заказа
df['diff_days'] = df.order_purchase_timestamp.apply(lambda x: (last_date - x).days)


# In[123]:


df


# In[128]:


# recency - насколько недавно была совершена последняя покупка клиента (количество дней с момента последней покупки минимальное):
df_recency = df.groupby('customer_unique_id', as_index = False) \
.agg({'diff_days' : 'min'}) \
.rename(columns = {'diff_days' : 'recency'})


# In[130]:


# frequency - как часто клиент совершает покупку (количество покупок за исследуемый период):
df_frequency = df.groupby('customer_unique_id', as_index = False) \
.agg({'order_id' : 'count'}) \
.rename(columns = {'order_id' : 'frequency'})


# In[141]:


# monetary - сколько денег тратит клиент (общая сумма покупок, сделанных за исследуемый период):
df_monetary = df.groupby('customer_unique_id', as_index = False) \
.agg({'price' : 'sum'}) \
.rename(columns = {'price' : 'monetary'})


# In[142]:


df_rec_freq = df_recency.merge(df_frequency)


# In[143]:


df_rfm = df_rec_freq.merge(df_monetary)


# In[144]:


df_rfm.head()


# Часто для качественного анализа аудитории использую подходы, основанные на сегментации. Используя python, построй RFM-сегментацию пользователей, чтобы качественно оценить свою аудиторию. В кластеризации можешь выбрать следующие метрики: R - время от последней покупки пользователя до текущей даты, F - суммарное количество покупок у пользователя за всё время, M - сумма покупок за всё время. Подробно опиши, как ты создавал кластеры. Для каждого RFM-сегмента построй границы метрик recency, frequency и monetary для интерпретации этих кластеров. Пример такого описания: RFM-сегмент 132 (recency=1, frequency=3, monetary=2) имеет границы метрик recency от 130 до 500 дней, frequency от 2 до 5 заказов в неделю, monetary от 1780 до 3560 рублей в неделю. (35 баллов)

# ### Исследуем метрики для определения кластеров

# In[138]:


df_rfm.recency.hist()


# In[147]:


df_rfm.recency.describe()


# recency - показатель удаленности последней покупки. Получается, что у 25% наших клиентов последний заказ был в течение последних трех месяцев, у 50% - в последние полгода, у 75% - в последние 7 месяцев, и у оставшихся - в первое полугодие. Видится три явных группы:  
# - 0,25 (полседние три месяца купили), 
# - 0,5 (покупка в последние полгода),  
# - и оставшиеся (покупка в первое полугодие).

# In[198]:


rec = df_rfm.recency.quantile([.25, .5]).to_dict()


# In[199]:


rec


# In[150]:


df_rfm.frequency.hist()


# In[153]:


df_rfm.frequency.describe()


# In[155]:


df_rfm.query('frequency > 1').value_counts('frequency')


# frequency - как часто клиент совершает покупку. Группы в разрезе frequency наблюдаются такими:  
# - 1 покупка (наибольшая группа),  
# - 2 покупки,  
# - 3 и более покупок

# In[195]:


fig = plt.figure(figsize = (7,7))
ax = fig.gca()
df_rfm.monetary.hist(ax = ax)


# In[156]:


df_rfm.monetary.describe()


# In[197]:


fig = plt.figure(figsize = (15,10))
df_rfm.boxplot('monetary')


# monetary - сколько денег тратит клиент. А клиент тратит по-разному:  
# - наибольшая группа до 152,  
# - вторая группа от 153 до 5000,
# - третья группа более 5000 (выбросы)
# Можно было бы и не делить между собой вторую и третью группы, но например это разделение можно использовать в программе лояльности, разные группы клиентов получают разные скидки(тут еще конечно надо смотреть остальные сегменты клиентов третьей группы)

# ### Расчет баллов RFM

# In[203]:


def r_score(x):
    if x <= rec[.25]:
        return 1
    elif x > rec[.5]:
        return 3
    else:
        return 2

def f_score(x):
    if x == 1:
        return 3
    elif x > 2:
        return 1
    else:
        return 2
    
def m_score(x):
    if x <= 152:
        return 3
    elif x > 5000:
        return 1
    else:
        return 2


# In[215]:


# внесем в датафрейм колонки со скорами:
df_rfm['r_score'] = df_rfm.recency.apply(lambda x: str(r_score(x)))
df_rfm['f_score'] = df_rfm.frequency.apply(lambda x: str(f_score(x)))
df_rfm['m_score'] = df_rfm.monetary.apply(lambda x: str(m_score(x)))


# In[219]:


# создадим колонку с указанием общего кластера
df_rfm['cluster'] = df_rfm.r_score + df_rfm.f_score + df_rfm.m_score


# In[225]:


df_rfm.cluster.value_counts()


# In[223]:


df_rfm.cluster.nunique()


# Итак, всего у нас получилось 20 кластеров. Большинство наших клиентов попали в RFM 333 (давнишние мелкие заказы). Также в топ-3 групп входят клиенты RFM 133 (мелкие заказы, сделанные в последние три месяца) и RFM 233 (мелкие заказы давности от 3 месяцев до полугода). Идеальная группа - RFM 111 (свежие крупные заказы в большом количестве), у нас она отсутствует вобще.
