from coinflip.classes import Mappers,FigureUtilities

from collections import Counter
from dataclasses import dataclass
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc
import os
import pandas as pd
from scipy.constants import convert_temperature
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import RidgeCV, LogisticRegressionCV, LassoCV, MultiTaskLassoCV, ElasticNetCV, MultiTaskElasticNetCV
from sklearn.svm import LinearSVR, SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import warnings


warnings.filterwarnings('ignore')

make_big = FigureUtilities.make_big
set_font_size = FigureUtilities.set_font_size
set_fig_size = FigureUtilities.set_fig_size

def read_training_data(infile='training_data.csv'):
    return pd.read_csv(infile).drop(columns=['Unnamed: 0']).sort_values(by='datetime').reset_index(drop=True)


def format_title(**kwargs):
    return ';'.join([ f'{k}={v}' for k,v in kwargs.items() ])

def do_plot(x,y,hue,data=None):
    if data is None:
            raise ValueError(f'Expected data:pd.DataFrame; got {data}:{type(data)}')
    dx_=data.copy()
    make_big()
    sns.boxplot(x=dx_[x],y=dx_[y],hue=dx_[hue])
    plt.xticks(rotation=70)
    plt.title(format_title(y=y, x=x, hue=hue))
    plt.legend(loc=4)
    plt.show()

def get_dt(dt):
    return datetime.datetime.fromisoformat(dt)
    
get_datetime = get_dt
    

def get_plots():
    
    df = read_training_data()
    set_font_size(size=10)
    set_font_size(size=18)
    set_fig_size()
    dd = df.drop(columns='atemp')
    sns.heatmap(dd.corr(), annot=True, cmap='coolwarm')

    corr = df.corr()
    set_fig_size()
    set_font_size()
    sns.heatmap(corr[['atemp','temp']], annot=True, cmap='coolwarm')
    plt.show()

    make_big()
    sns.boxplot(x='atemp', y='temp', data=df)
    plt.xticks(rotation=70)
    
    make_big()
    sns.pairplot(df, hue='season')
    
    df = df.sort_values(by='datetime').reset_index(drop=True)
    new_cols = list(df.columns[1:])+[df.columns[0]]
    dd = df[new_cols]
    # print(new_cols)
    dd = dd.drop(columns='atemp')
    # print('dropping atemp',dd)


    dx = dd.copy()

    season_mapper = {
        1: 'spring',
        2: 'summer',
        3: 'fall',
        4: 'winter'
    }

    weekday_mapper = {
        0: 'monday',
        1: 'tuesday',
        2: 'wednesday',
        3: 'thurday',
        4: 'friday',
        5: 'saturday',
        6: 'sunday'

    }

    def temp_mapper(temp=None, 
                    low=12, 
                    high=29):

        if temp is None:
            raise ValueError(f'temp must be a number. got {temp}')

        if temp<low//2:
            return 'freezing'
        elif temp<low:
            return 'cold'
        elif temp>=high*1.33:
            return 'hades'
        elif temp <= high:
            return 'nice'
        else:
            return 'hot'
    
    dx['tempcat'] = list(map(temp_mapper, dx.temp))
    dts = df['datetime'].copy()
    dx['hour'] = dts.apply(func= lambda x: get_datetime(x).hour)
    dx['weekday'] = dts.apply(func= lambda x: get_datetime(x).weekday())
    dx['weekday'] = dx.weekday.apply(func=lambda x: weekday_mapper[x])
    dx['year'] = dts.apply(func= lambda x: (get_datetime(x).year - 2011)%2011+1)
    dx['month'] = dts.apply(func=lambda x: get_datetime(x).month )
    dx['approx_temp'] = df.temp.apply(func=lambda x: round(x,1))
    low_temp_F, high_temp_F=49,78
    dx['temp_cat'] = dx.temp.apply(func=lambda x : temp_mapper(temp=x,low=convert_temperature(low_temp_F,'f','c'),high=convert_temperature(high_temp_F,'f','c')))
    dx['season'] = dx.season.apply(func=lambda x: season_mapper[x])
    return dx


def get_dx(infile='training_data.csv', verbose=False):
    verbose and print(f'reading {infile}')
    df = read_training_data(infile=infile)
    df = df.sort_values(by='datetime').reset_index(drop=True)
    new_cols = list(df.columns[1:])+[df.columns[0]]
    dd = df[new_cols]
    dd = dd.drop(columns='atemp')
    dx = dd.copy()
    
    def get_dt(dt):
        return datetime.datetime.fromisoformat(dt)

    season_mapper = Mappers.season_mapper

    weekday_mapper = {
        0: 'monday',
        1: 'tuesday',
        2: 'wednesday',
        3: 'thurday',
        4: 'friday',
        5: 'saturday',
        6: 'sunday'

    }

    def temp_mapper(temp=None, 
                    low=12, 
                    high=29):

        if temp is None:
            raise ValueError(f'temp must be a number. got {temp}')

        if temp<low//2:
            return 'freezing'
        elif temp<low:
            return 'cold'
        elif temp>=high*1.33:
            return 'hades'
        elif temp <= high:
            return 'nice'
        else:
            return 'hot'

    dx['tempcat'] = list(map(temp_mapper, dx.temp))
    dts = df['datetime'].copy()
    dx['hour'] = dts.apply(func= lambda x: get_datetime(x).hour)
    dx['weekday'] = dts.apply(func= lambda x: get_datetime(x).weekday())
    dx['weekday'] = dx.weekday.apply(func=lambda x: weekday_mapper[x])
    dx['year'] = dts.apply(func= lambda x: (get_datetime(x).year - 2011)%2011+1)
    dx['month'] = dts.apply(func=lambda x: get_datetime(x).month )
    dx['temp_cat'] = dx.temp.apply(func=lambda x : temp_mapper(temp=x))
    dx['season'] = dx.season.apply(func=lambda x: season_mapper[x])
    return dx

get_X = get_dx