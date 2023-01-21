from coinflip.classes import FigureUtilities, Mappers
from collections import Counter
import datetime
from matplotlib import rc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pprint import pprint
import seaborn as sns
from scipy.constants import convert_temperature
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV,  ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, PolynomialFeatures, Normalizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.svm import LinearSVR
from time import time
import warnings


def set_fig_size(width=18,height=10):
    fig,ax=plt.subplots(1,1)
    fig.set_size_inches(width,height)
    return fig,ax
    
def set_font_size(size=18):
    font = {'family' : 'verdana',
            'size'   : size}
    rc('font', **font)
    
def make_big(f=18,w=18,h=10):
    set_font_size(size=f)
    fig, ax = set_fig_size(width=w,height=h)
    return fig, ax

def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def get_df(drop_index=False, sort=False):
    df = pd.read_csv('training_data.csv').drop(columns=['Unnamed: 0'])
    if sort:
        df = df.sort_values(by='datetime')
    if drop_index:
        return df.reset_index(drop=True)
    return df

def get_dd(drop_index=True, sort=True):
    df = get_df(drop_index=drop_index, sort=sort).drop(columns=['atemp'])
    return df[list(df.columns[1:])+[df.columns[0]]]

def do_little_eda():
    df = get_df()
    make_big()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    corr = df.corr()
    
    make_big()
    sns.heatmap(corr[['atemp','temp']], annot=True, cmap='coolwarm')
    plt.show()
    
    make_big()
    sns.boxplot(x='atemp', y='temp', data=df)
    plt.xticks(rotation=70)
    
    make_big()
    sns.pairplot(df, hue='season')
    
def get_dt(dt):
    return datetime.datetime.fromisoformat(dt)
#f(100, n=5, m=2) --> 40,60

def get_slice(data, 
              n_slices=5, 
              get_sliceno=1,
              debug=False,
              indices=False):
    '''
    def get_slice(
                data: Iterable[Any], 
                n_slices=5, 
                get_sliceno=1,
                debug=False,
                indeces=False)
                
    
    Get the Mth partition of a timeseries of values out of N slices, where (strictly) M < N.
    
    :param data: the data (iterable) that is to be partitioned.
    :param n_slices=5: the number of slices into which to partition the data.
    :param get_sliceno=1: let get_sliceno=M, n_slices=N, then the function provides the Mth of N slices
        (zero indexed! that is, the 0th slice is the slice with the lowest indes, and the N-1th slice
        has the largest indices.
    :param debug=False: if on, the function returns index and values
    :param index=False: if on, the function returns the indices resulting slice, rather than
        the values.  (Note however, debug takes precedence, so debug==index==1 is the same as debug=1)
    
    Example: Imagine you have a 100cm meter stick. We want to get the M=2nd slice of n_slices=5:
    
    Calling `get_slice(ruler_data, n_slices=5, get_slice=4)` partitions it into 5 slices, making ruler looks like
    
                edge of   |       |       |       |       |        |
                   ruler->|   |   |   |   |   |   |   |   |   |    |<-endge of
                         0| 10| 20| 30| 40| 50| 60| 70| 80| 90| 100|      ruler
                          |   |   |   |   |   |   |   |   |   |    |
                          |       |       |       |       |        |

                                         partition_number
                             0th     1st     2nd     3rd     4th
                                          ^       ^
                                          |       |
                                          |result | 
                                          | slice |
                                          |       |
                                      ##################
                                      #   40      60   #
                                      ##################
                                      

   >> ruler_data = np.linspace(0,100,101)
   >> print('ruler','\t',ruler_data)
   >> print()
   >> print('2nd of 5)
   >> print('slices')
   >> print( 
            get_slice(ruler_data, 
                      n_slices=5, 
                      get_sliceno=2)
        )
    
    
    entire ruler	array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                            13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
                            26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                            39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
                            52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
                            65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
                            78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
                            91,  92,  93,  94,  95,  96,  97,  98,  99, 100])
                            
    2nd of 5 slices	  [40. 41. 42. 43. 44. 45. 46. 47. 48. 49. 50. 51. 52. 53. 54. 55. 56. 57. 58. 59.]   
    
    
    >> get_slice(np.linspace(0,100, 101), n_slices=10, get_sliceno=1)
    array([50., 51., 52., 53., 54., 55., 56., 57., 58., 59.])
    
    >> import random
    >> from numpy import array
    >> Amn = array([ [ random.randint(-1000,1000) for i in range(10) ] for j in range(100)])
    >> data = pd.DataFrame(columns=list('abcdefghij'), data=Amn)
    >> get_slice(data, n_slices=6, get_sliceno=2)
    
          a    b    c    d    e    f    g    h    i    j
    32 -188  924 -777  298  407 -620  289  515 -893   61
    33 -725 -574 -478 -454 -415 -325 -935  775  298  765
    34 -659  239 -872  673  652 -808  641  169 -596  242
    35  234  333 -533 -805  263   22  298  889  374  839
    36  900 -950 -619 -375  359 -781   25  638  357  120
    37  502  647 -369 -719   -9 -677 -380 -806  800  771
    38  350  902   24  282 -651  -39  316 -730  939  702
    39  185 -461  720  189 -887  739 -933 -981  795 -788
    40 -585  800   27  711 -925 -505  -58  638  496 -790
    41  678  792 -454  266 -644  731 -577  508  536 -858
    42  913  640  645 -192 -213 -784    4  -37 -772  471
    43   93  632  468  -70  528  972  339 -208 -725  847
    44  827 -339 -159 -597  321 -221 -972 -929  117 -886
    45  629 -524  337  135 -122  -73 -253 -660  -93   84
    46 -314  190   37  606  -86  414 -679  494 -128  441
    47  988 -137  451  658  565  276 -713 -388  851  859

    '''
    width = len(data)//n_slices
    end_position = width*(get_sliceno+1)
    start_position = end_position - width
    return data[start_position:end_position]
    
    

    
@timer_func
def get_dx(verbose=False):
    '''
    Reads input file and prepares data for the pipeline.
    '''
    
    def do_verbose(verbose=verbose):
        if verbose:
            print(dx[['datetime','year','hour','month','hr_season','yr_month','yr_season']])
            print()
            print('feature name,','number of disctint values')
            print('set of distinct values')
            print('='*80)
            for time_var in [#'datetime',
                             'season',
                             'year',
                             'hour',
                             'month',
                             'hr_season',
                             'yr_month',
                             'yr_season']:
                print()
                print(time_var,len(list(sorted(list(set(dx[time_var]))))))
                print(list(sorted(list(set(dx[time_var])))))
                    
    low_temp_F, high_temp_F=49,78
    dx = get_dd()
    
    get_datetime   = get_dt
    weekday_mapper = Mappers.weekday_mapper
    season_mapper  = Mappers.season_mapper
    temp_mapper    = Mappers.temp_mapper
    fromisoformat  = datetime.datetime.fromisoformat
    
    dx['tempcat'] = list(map(Mappers.temp_mapper, dx.temp))
    dts = dx['datetime'].copy()
    dx['weekday'] = dts.apply(func=lambda x: weekday_mapper(get_datetime(x).weekday()))
    dx['year'] = dts.apply(func= lambda x: (get_datetime(x).year - 2011)%2011+1)
    dx['month'] = dts.apply(func=lambda x: get_datetime(x).month )
    dx['hour'] = [ get_dt(x).hour for x in dx.datetime ]
    dx['season'] = dx.season.apply(func=lambda x: Mappers.season_mapper(x))
    dx['temp_cat'] = dx.temp.apply(func=lambda x : Mappers.temp_mapper(temp=x,
                                                                       low=convert_temperature(low_temp_F,'f','c'),
                                                                       high=convert_temperature(high_temp_F,'f','c')))
    # 2nd degree interactions
    dx['hr_workingday'] = [ str(dx.iloc[idx].hour) + ','+str(dx.iloc[idx].workingday) for idx in range(len(dx))]
    dx['hr_tempcat'] = [ str(dx.iloc[idx].hour) + ','+str(dx.iloc[idx].tempcat) for idx in range(len(dx))]                                                                         
    dx['hr_season'] = [ str(dx.iloc[idx].hour) + ','+str(dx.iloc[idx].season) for idx in range(len(dx))]
    dx['hr_month'] = [ str(dx.iloc[idx].hour) + ',' + str(dx.iloc[idx].month) for idx in range(len(dx))]
    dx['hr_weekday'] = [ str(dx.iloc[idx].hour) + ',' +str(dx.iloc[idx].weekday) for idx in range(len(dx))]
    dx['yr_month'] = [ str(get_dt(x).year) + ','+str(get_dt(x).month) for x in dx.datetime ]
    dx['yr_season'] = [ str(get_dt(dx.iloc[idx].datetime).year) + ','+str(dx.iloc[idx].season) for idx in range(len(dx)) ]
    dx['season_weekday'] = [ ','.join(x.split(',')[1:]) for x in dx.yr_season_weekday ]
    # 3rd degree interations
    dx['hr_weekday_season'] =  [ str(dx.iloc[idx].hr_weekday)+','+str(dx.iloc[idx].season) for idx in range(len(dx))]
    dx['hr_workingday_season'] = [ str(dx.iloc[idx].hr_workingday)+','+str(dx.iloc[idx].season) for idx in range(len(dx)) ]
    dx['yr_season_weekday']= [ row.yr_season+','+str(row.weekday) for idx, row in dx.iterrows() ] 
    # masive 4th order vairbale
    dx['yr_season_weekday_hr'] = [ row.yr_season_weekday+','+str(row.hour) for idx,row in dx.iterrows() ]
    return dx.reset_index(drop=True)
    
def get_dx_new(verbose=False):
    return get_dx(verbose=verbose)


@timer_func
def main(pipelines=None,
         dep_var='casual',
         data=None):
    
    if data is None:
        data = get_dx_new()
    
    y = data[['casual','registered','count']]
    data = data.drop(columns=list(y.columns))
    y = y[dep_var]
    X = pd.get_dummies(data)
    
    if pipelines is None:
        pipelines=dict()

    fig, ax = make_big()
    
    models = Mappers._model_mapper
    
    for modelname in Mappers._model_mapper.keys():
        label = modelname+','+dep_var
        model = models[modelname]()
        pipeline = make_pipeline(MinMaxScaler(), model)
        
        X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        pipeline.fit(X_train, y_train)
        y_predict = pipeline.predict(X_test)

        maxx = max(max(y_true),max(y_true))
        pipeline.maxx = maxx
        pipelines[label] = pipeline
        plt.scatter(y_true,y_predict, label=label)

    plt.plot(range(maxx),range(maxx), color='k',linewidth=3, label='y_true=y_pred')
    plt.title(f'Results')
    plt.xlabel('y_true')
    plt.ylabel('y_predict')
    plt.legend()
    plt.show()

    return pipelines


for dep_var in ['casual','registred']:
    pipes['dep_var'] = main()
    
