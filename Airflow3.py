#!/usr/bin/env python
# coding: utf-8

# In[1]:


# определение варианта (2007 год)
1994 + hash(f'e-malofeeva-22') % 23

import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime

from airflow import DAG
from airflow.decorators import dag, task

default_args = {
    'owner': 'e.malofeeva',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2022, 7, 31),
    'schedule_interval' : '0 8 * * *'
}


@dag(default_args = default_args, catchup = False)
def e_malofeeva_airflow3():
    # таска считывания файла
    @task()
    def get_data():
        df = pd.read_csv('vgsales.csv')
        df = df.query('Year == 2007.0')
        return df.to_csv('df.csv')

    # Какая игра была самой продаваемой во всем мире?
    @task()
    def most_popular_name(df):
        df = pd.read_csv('df.csv')
        most_popular_name = df.groupby('Name', as_index = False) \
        .agg({'Global_Sales' : 'sum'}) \
        .sort_values('Global_Sales', ascending = False) \
        .iloc[0]
        return most_popular_name.to_csv('most_popular_name.csv', index = False)

    # Игры какого жанра были самыми продаваемыми в Европе? Перечислить все, если их несколько
    @task()
    def most_popular_genre_eu(df):
        df = pd.read_csv('df.csv')
        most_popular_genre_eu = df.groupby('Genre', as_index = False) \
        .agg({'EU_Sales' : 'sum'}) \
        .sort_values('EU_Sales', ascending = False) \
        .iloc[0]
        return most_popular_genre_eu.to_csv('most_popular_genre_eu.csv', index = False)

    # На какой платформе было больше всего игр, которые продались более чем миллионным тиражом в Северной Америке?
    # Перечислить все, если их несколько
    @task()
    def most_popular_platform_na(df):
        df = pd.read_csv('df.csv')
        most_popular_platform_na = df.groupby('Platform', as_index = False) \
        .agg({'NA_Sales' : 'sum'}) \
        .sort_values('NA_Sales', ascending = False) \
        .query('NA_Sales > 1')
        return most_popular_platform_na.to_csv('most_popular_platform_na.csv', index = False)

    # У какого издателя самые высокие средние продажи в Японии? Перечислить все, если их несколько
    @task()
    def most_popular_publisher_jp(df):
        df = pd.read_csv('df.csv')
        most_popular_publisher_jp = df.groupby('Publisher', as_index = False) \
        .agg({'JP_Sales' : 'mean'}) \
        .sort_values('JP_Sales', ascending = False) \
        .iloc[0]
        return most_popular_publisher_jp.to_csv('most_popular_publisher_jp.csv ', index = False)

    # Сколько игр продались лучше в Европе, чем в Японии?
    @task()
    def df_sales_jp_eu(df):
        df = pd.read_csv('df.csv')
        df_sales_jp = df.groupby('Name', as_index = False) \
        .agg({'JP_Sales' : 'sum'})
        df_sales_eu = df.groupby('Name', as_index = False) \
        .agg({'EU_Sales' : 'sum'})
        df_sales_jp_eu = df_sales_jp.merge(df_sales_eu)
        df_sales_jp_eu['diff_sales'] = df_sales_jp_eu.EU_Sales - df_sales_jp_eu.JP_Sales
        games_eu = df_sales_jp_eu.query('diff_sales > 0').shape[0]
        return games_eu

    @task()
    def print_data(most_popular_name, most_popular_genre_eu, most_popular_platform_na, most_popular_publisher_jp, games_eu):
        #date = ''

        print(f'Most popular game')
        print(most_popular_name)

        print(f'Most popular genre in Europe')
        print(most_popular_genre_eu)

        print(f'Most popular  plarform in USA')
        print(most_popular_platform_na)

        print(f'Most popular publisher in Japan')
        print(most_popular_publisher_jp)

        print(f'Games with sales in Europe better than in Japan')
        print(games_eu)
        
    df = get_data()
    most_popular_name = most_popular_name(df)
    most_popular_genre_eu = most_popular_genre_eu(df)
    most_popular_platform_na = most_popular_platform_na(df)
    most_popular_publisher_jp = most_popular_publisher_jp(df)
    games_eu = df_sales_jp_eu(df)
    print_data(most_popular_name, most_popular_genre_eu, most_popular_platform_na, most_popular_publisher_jp, games_eu)   
    
e_malofeeva_airflow3 = e_malofeeva_airflow3()

