from collections import Counter
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc
import numpy as np
import os
import pandas as pd
from scipy.constants import convert_temperature
import seaborn as sns


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, LassoCV,  ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR, SVR


from pprint import pprint
import warnings


SHOW_ALL_PLOTS=False

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


def get_df(reset_index_and_drop=False):
    df = pd.read_csv('training_data.csv').drop(columns=['Unnamed: 0'])
    if reset_index_and_drop:
        return df.reset_index(drop=True)
    return df

df = get_df()

def get_dd():
    df = pd.read_csv('training_data.csv').drop(columns=['Unnamed: 0'])
    dd = df.drop(columns='atemp')

    df = df.sort_values(by='datetime').reset_index(drop=True)
    new_cols = list(df.columns[1:])+[df.columns[0]]
    dd = df[new_cols]
    return dd

dd = get_dd()


def do_little_eda():
    df=get_df()
    print(df.corr())

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
    


def get_dt(dt):
    return datetime.datetime.fromisoformat(dt)



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
    

def get_dx_old():
    get_datetime = get_dt
    from scipy.constants import convert_temperature
    dx = dd.copy()
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

dx = get_dx_old()

# plotting helper
def format_title(**kwargs):
    return ';'.join([ f'{k}={v}' for k,v in kwargs.items() ])
    
def do_plot(x,y,hue,data=dx):
    dx_=data.copy()
    fig, ax = make_big()
    sns.boxplot(x=dx_[x],y=dx_[y],hue=dx_[hue])
    plt.xticks(rotation=70)
    plt.title(format_title(y=y, x=x, hue=hue))
    plt.legend(loc=4)
    plt.show()


def read_input(infile):    
    return pd.read_csv(infile).drop(columns='Unnamed: 0').sort_values(by='datetime').reset_index(drop=True)


# first stage
def process_input(infile, 
                  kind='train',
                  verbose=False):
    print('Processing:', infile, 'kind:', kind, 'filename:', infile)
                                  
    if kind not in ('train', 'test'):
        raise ValueError(f'kind must be set to either "train" or "test"; got {kind}')
        
    
    def get_dt(dt):
        return datetime.datetime.fromisoformat(dt)
    low_temp_F, high_temp_F=49,78
    data = read_input(infile)
    dd = data.drop(columns='atemp')
    dx = dd.copy()
    from scipy.constants import convert_temperature
    dx = dd.copy()
    dx['tempcat'] = list(map(temp_mapper, dx.temp))
    dts = df['datetime'].copy()
    # dx['hour'] = dts.apply(func= lambda x: get_datetime(x).hour)
    dx['weekday'] = dts.apply(func= lambda x: get_datetime(x).weekday())
    dx['weekday'] = dx.weekday.apply(func=lambda x: weekday_mapper[x])
    dx['year'] = dts.apply(func= lambda x: (get_datetime(x).year - 2011)%2011+1)
    dx['month'] = dts.apply(func=lambda x: get_datetime(x).month )
    dx['approx_temp'] = df.temp.apply(func=lambda x: round(x,1))
    dx['temp_cat'] = dx.temp.apply(func=lambda x : temp_mapper(temp=x,low=convert_temperature(low_temp_F,'f','c'),high=convert_temperature(high_temp_F,'f','c')))
    dx['season'] = dx.season.apply(func=lambda x: season_mapper[x])
    dx['hour'] = [ str(get_dt(x).time().hour) for x in dx.datetime ]
    #dx['month'] = [ str(get_dt(x).month) for x in dx.datetime ]
    #dx['hr_season'] = [ str(get_dt(dx.iloc[idx].datetime).time().hour) + ','+str(dx.iloc[idx].season) for idx in range(len(dx))]
    #dx['yr_month'] = [ str(get_dt(x).year) + ','+str(get_dt(x).month) for x in dx.datetime ]
    #dx['yr_season'] = [ str(get_dt(dx.iloc[idx].datetime).year) + ','+str(dd.iloc[idx].season) for idx in range(len(dx)) ]
    dx['hr_workingday'] = [ str(get_dt(dx.iloc[idx].datetime).time().hour) + ','+str(dx.iloc[idx].workingday) for idx in range(len(dx))]
    dx['hr_workingday_month'] = [str(get_dt(dx.iloc[idx].datetime).time().hour) + ','+str(dx.iloc[idx].workingday)+','+str(dx.iloc[idx].month) for idx in range(len(dx))]
    dx['hr_tempcat'] = [ str(get_dt(dx.iloc[idx].datetime).time().hour) + ','+str(dx.iloc[idx].tempcat) for idx in range(len(dx))]                                                                         
    dx['hr_season'] = [ str(get_dt(dx.iloc[idx].datetime).time().hour) + ','+str(dx.iloc[idx].season) for idx in range(len(dx))]
    dx['yr_month'] = [ str(get_dt(x).year) + ','+str(get_dt(x).month) for x in dx.datetime ]
    dx['yr_season'] = [ str(get_dt(dx.iloc[idx].datetime).year) + ','+str(dx.iloc[idx].season) for idx in range(len(dx)) ]
    dx['weekday'] = [ weekday_mapper[datetime.datetime.fromisoformat(dx.iloc[idx].datetime).weekday()] for idx in range(len(dx)) ]
    dx['yr_season_weekday']= [ row.yr_season+','+str(row.weekday) for idx, row in dx.iterrows() ] 
    dx['season_weekday'] = [ ','.join(x.split(',')[1:]) for x in dx.yr_season_weekday ]
    dx['hr_weekday'] = [ ','.join(list(map(str,[row.hour,row.weekday]))) for idx, row in dx.iterrows() ]
    dx['yr_season_weekday_hr'] = [ row.yr_season_weekday+','+str(row.hour) for idx,row in dx.iterrows() ]
    dx['season_weekday_hr'] =  [ ','.join(x.split(',')[1:]) for x in dx['yr_season_weekday_hr'] ]
    #dx['workday_hr'] = [ ','.join([str(row.workingday), str(row.hour)]) for idx, row in dx.iterrows() ]


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


    dx = dx.reset_index(drop=True)

    # print('HEY!',dx.columns)
    if kind == 'train':
        y_vars = [
            'casual', 
            'registered', 
            'count']
    elif kind == 'test':
        y_vars = []

    x_continuous = [
            'temp',
            'humidity',
            'windspeed']

    drop_timevars = [
            'datetime',
            'index',
            'season',
            'year',
            'hour',
            'month']

    x_categorical = [ col for col in dx.columns if col not in y_vars+x_continuous+drop_timevars ] 

    if verbose:
        print('Y:', y_vars)
        print('X_cont:', x_continuous)
        print('X_cat:', x_categorical)
        print('All:', dx.columns)

    d_y = dx[y_vars]
    d_xcat = dx[x_categorical]
    d_xcont = dx[x_continuous]

    
    print('kind:', kind)
    if kind == 'train':
        verbose and print('return d_y, d_xcat, d_xcont')
        return d_y, d_xcat, d_xcont
    else:
        verbose and print('return d_xcat, d_xcont')
        return d_xcat, d_xcont

# helper
def copy_columns(to_=None, from_=None):
    
    _to, _from= to_.copy(),from_.copy()
    
    if to_ is None:
        raise ValueError(f'Set to_ to a destination DataFrame. Got {to_}')
    elif from_ is None:
        raise ValueError(f'Set from_ to a source DataFrame. Got {from_}')
    
    for colname in from_.columns:
        _to[colname] = from_[colname]
        
    return _to


# preprocessors
def get_vectorized(data,
                   sparse=False,
                   filter_data=False):
    if filter_data:
        data = data[filter_data]
        
    v = DictVectorizer(sparse=sparse)
    return v, v.fit_transform(data.to_dict('records'))
def get_dict_vectorized(data, sparse=False, filter_data=False):
    if filter_data:
        data = data[filter_data]
    return DictVectorizer(sparse=sparse).fit_transform(data.to_dict('records'))

def get_minmax_scaled(data:pd.DataFrame):
    data = data.copy()
    scaler = MinMaxScaler()
    return scaler, pd.DataFrame(columns = data.columns,
                 data = scaler.fit_transform(data.to_numpy()))





def get_dummies(data, drop_first=False, filter_data=False):
    if filter_data:
        data = data[filter_data]
    return pd.get_dummies(data, drop_first=drop_first)




def get_x_train(infile='training_data.csv'):
    '''
    def get_x_train(inp='training_data.csv')->np.array
    '''
    train_y, train_xcat, train_xcont = process_input(infile=infile,
                                                     kind='train')
    vectorizer,xvectorized = get_vectorized(train_xcat)
    scaler, xscaled = get_minmax_scaled(data=train_xcont)
    return np.concatenate((xvectorized, xscaled), 
                          axis=1)

def get_x_test(infile='test_data.csv'):
    test_xcat, test_xcont = process_input('test_data.csv',
                                      kind='test')
    vectorizer,xvectorized = get_vectorized(test_xcat)
    scaler, xscaled = get_minmax_scaled(data=test_xcont)
    return np.concatenate((xvectorized, xscaled), 
                          axis=1)
    

def get_y_train(infile='training_data.csv',
                minmaxscale=False):
    
    train_y, train_xcat, train_xcont = process_input(infile=infile,
                                                     kind='train')
    return train_y


def get_train_test_split(minmax_y=False,
                         shuffle=False):
    
    X_TRAIN = get_x_train()
    Y_TRAIN = get_y_train()
    print(X_TRAIN.shape)
    print(Y_TRAIN.shape)

    X_TEST = get_x_test()
    print(X_TEST.shape)

    if minmax_y:
        yscaler, y = get_minmax_scaled(Y_TRAIN)
    else:
        y_scaler, y = None, Y_TRAIN
    X = X_TRAIN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=shuffle)

    if minmax_y:
        return X_train, X_test, y_train, y_test, y_scaler
    
    return X_train, X_test, y_train, y_test




def make_models0(minmax_y=False,
                preserve_order=False):
    '''
    Uses local vars to produce an output dataframe.
    References pre-defined train/test split.
    '''
    
    _records_dict, _records = dict(), []

    shuffle=not(preserve_order)
    
    if not minmax_y:
        X_train, X_test, y_train, y_test = get_train_test_split(minmax_y=minmax_y, shuffle=shuffle)
    else:
        X_train, X_test, y_train, y_test, y_scaler = get_train_test_split(minmax_y=minmax_y, shuffle=shuffle)

    outputs,model_outputs = [],dict()
    ar = np.array
    models = {
        'ridge': RidgeCV, 
        #'svm lin': LinearSVR,
        #'svm': SVR,
       # 'linear_reg': LinearRegression,
        'elasticnet': ElasticNetCV
    }
    
    dep_vars = y_test.columns

    
    for dep_var in ['casual', 'registered']:
        for modelname, m in models.items():
            label = modelname+','+dep_var
            model = m()
         
            if modelname == 'lasso':
                model = LassoCV()
            elif modelname == 'elasticnet':
                model = ElasticNetCV()

            y_true = y_test[dep_var]
            y_tr = y_train[dep_var]

            try:
         
                print(f'fitting {label}')
                model.fit(X_train, y_tr)
                y_pred = model.predict(X_test)

                ys = y_true, y_pred
                
                e1,e2,e3,e4 = mean_absolute_error(*ys),mean_absolute_percentage_error(*ys),mean_squared_error(*ys),np.nan#ar([_**0.5 for _ in mean_squared_error(*ys)])

                #e1,e2,e3,e4 = mean_absolute_error(*ys),mean_absolute_percentage_error(*ys),mean_squared_error(*ys),mean_squared_log_error(*ys)#ar([_**0.5 for _ in mean_squared_error(*ys)])

                # label = modelname+','+dep_var

                try:
                    coefs = model.coef_
                except:
                    coefs = np.nan

                if modelname == 'svm':
                    coefs = np.nan


                this_output = {'label': label,
                               'modelname': modelname,
                               'model': model,
                               'dep_var': dep_var,
                               'strmodel': str(model),
                               'coefs': coefs,
                               'mean_absolute_error': float(e1),
                               'mean_absolute_percentage_error': float(e2),
                               'mean_squared_error': float(e3),
                               'mean_squared_log_error': float(e4),
                               'y_true': y_true,
                               'y_pred': y_pred,
                               'error': ''}

                # raw_model_outputs[modelname+','+dep_var] = (results)
                _records_dict[label]=this_output
                _records.append(this_output)

                # print(this_output)
    #             print(*results)
            except:
                print(sys.exc_info())
                keys_ = ['modelname',
                         'model',
                         'dep_var',
                         'strmodel',
                         'coefs',
                         'mean_absolute_error',
                         'mean_absolute_percentage_error',
                         'mean_squared_error',
                         'y_true',
                         'y_pred',
                         'error']
                this_output = { _:np.nan for _ in keys_ }
                this_output['error'] = str(sys.exc_info())
                _records_dict[label] = this_output
                _records.append(this_output)
                
    dm = pd.DataFrame.from_records(_records)
    dm['y_true']=dm.y_true.apply(lambda x: x.values)
    dm['ytr']=dm.y_true.apply(lambda x: ','.join(list(map(str, x))))
    dm['y_pred']=dm.y_pred.apply(lambda x: np.array(list(map(lambda y: round(y,1), x))))
    dm['ypr']=dm.y_pred.apply(lambda x: ','.join(list(map(str, x))))

    return dm, _records_dict, _records


def get_dm_0(verbose=False):
    dm, model_outputs, records = make_models(preserve_order=True)
    verbose and print(dm.columns)
    dm['rms_error'] = dm.mean_squared_error
    dm = dm[~pd.isnull(dm.rms_error)]
    errors = dm[['model','rms_error', 'mean_absolute_error','mean_squared_error']]
    errors.rms_error = [ np.sqrt(e) for e in errors.rms_error ]
    verbose and print(errors.head(4))
    verbose and print(dm.head(4))
    return dm

def make_results_scatter0(dm=None,
                         verbose=False):
    if dm is None:
        dm = get_dm(verbose=verbose)
        
    for didx,dep_var in enumerate(['registered', 'casual']):
        fix, ax = set_fig_size(14,9)
        set_font_size(20)
        N=0
        d=dm[dm.dep_var==dep_var]
        maxx = max(int(max(np.concatenate(d.y_true.values))),
                   int(max(np.concatenate(d.y_pred.values))))

        for idx, row in d.iterrows():
            ax.scatter(row.y_true, row.y_pred, label=row.label, marker='o')
            N+-1

        ax.plot(range(maxx), range(maxx), linewidth=3, color='k')
        plt.xlabel('y_true')
        plt.ylabel('y_pred')
        plt.title(f'Results, {dep_var}')
        plt.legend()
        plt.savefig(f'res{didx}.png')
        plt.show()


def make_models(minmax_y=False,
                preserve_order=False):
    '''
    Uses local vars to produce an output dataframe.
    References pre-defined train/test split.
    '''
    
    _records_dict, _records = dict(), []

    shuffle=not(preserve_order)
    
    if not minmax_y:
        X_train, X_test, y_train, y_test = get_train_test_split(minmax_y=minmax_y, shuffle=shuffle)
    else:
        X_train, X_test, y_train, y_test, y_scaler = get_train_test_split(minmax_y=minmax_y, shuffle=shuffle)

    outputs,model_outputs = [],dict()
    ar = np.array
    models = {
        'ridge': RidgeCV, 
        #'svm lin': LinearSVR,
        #'svm': SVR,
       # 'linear_reg': LinearRegression,
        'elasticnet': ElasticNetCV
    }
    
    dep_vars = y_test.columns

    
    for dep_var in ['casual', 'registered']:
        for modelname, m in models.items():
            label = modelname+','+dep_var
            model = m()
         
            if modelname == 'lasso':
                model = LassoCV()
            elif modelname == 'elasticnet':
                model = ElasticNetCV()

            y_true = y_test[dep_var]
            y_tr = y_train[dep_var]

            try:
         
                print(f'fitting {label}')
                model.fit(X_train, y_tr)
                y_pred = model.predict(X_test)

                ys = y_true, y_pred
                
                e1,e2,e3,e4 = mean_absolute_error(*ys),mean_absolute_percentage_error(*ys),mean_squared_error(*ys),np.nan#ar([_**0.5 for _ in mean_squared_error(*ys)])

                #e1,e2,e3,e4 = mean_absolute_error(*ys),mean_absolute_percentage_error(*ys),mean_squared_error(*ys),mean_squared_log_error(*ys)#ar([_**0.5 for _ in mean_squared_error(*ys)])

                # label = modelname+','+dep_var

                try:
                    coefs = model.coef_
                except:
                    coefs = np.nan

                if modelname == 'svm':
                    coefs = np.nan


                this_output = {'label': label,
                               'modelname': modelname,
                               'model': model,
                               'dep_var': dep_var,
                               'strmodel': str(model),
                               'coefs': coefs,
                               'mean_absolute_error': float(e1),
                               'mean_absolute_percentage_error': float(e2),
                               'mean_squared_error': float(e3),
                               'mean_squared_log_error': float(e4),
                               'y_true': y_true,
                               'y_pred': y_pred,
                               'error': ''}

                # raw_model_outputs[modelname+','+dep_var] = (results)
                _records_dict[label]=this_output
                _records.append(this_output)

                # print(this_output)
    #             print(*results)
            except:
                print(sys.exc_info())
                keys_ = ['modelname',
                         'model',
                         'dep_var',
                         'strmodel',
                         'coefs',
                         'mean_absolute_error',
                         'mean_absolute_percentage_error',
                         'mean_squared_error',
                         'y_true',
                         'y_pred',
                         'error']
                this_output = { _:np.nan for _ in keys_ }
                this_output['error'] = str(sys.exc_info())
                _records_dict[label] = this_output
                _records.append(this_output)
                
    dm = pd.DataFrame.from_records(_records)
    dm['y_true']=dm.y_true.apply(lambda x: x.values)
    dm['ytr']=dm.y_true.apply(lambda x: ','.join(list(map(str, x))))
    dm['y_pred']=dm.y_pred.apply(lambda x: np.array(list(map(lambda y: round(y,1), x))))
    dm['ypr']=dm.y_pred.apply(lambda x: ','.join(list(map(str, x))))

    return dm, _records_dict, _records
    
    
def get_dm(verbose=False):
    dm, model_outputs, records = make_models(preserve_order=True)
    verbose and print(dm.columns)
    dm['rms_error'] = dm.mean_squared_error
    dm = dm[~pd.isnull(dm.rms_error)]
    errors = dm[['model','rms_error', 'mean_absolute_error','mean_squared_error']]
    errors.rms_error = [ np.sqrt(e) for e in errors.rms_error ]
    verbose and print(errors.head(4))
    verbose and print(dm.head(4))
    return dm


def make_results_scatter(dm=None,
                         verbose=False):
    if dm is None:
        dm = get_dm(verbose=verbose)
        
    for didx,dep_var in enumerate(['registered', 'casual']):
        fix, ax = set_fig_size(14,9)
        set_font_size(20)
        N=0
        d=dm[dm.dep_var==dep_var]
        maxx = max(int(max(np.concatenate(d.y_true.values))),
                   int(max(np.concatenate(d.y_pred.values))))

        for idx, row in d.iterrows():
            ax.scatter(row.y_true, row.y_pred, label=row.label, marker='o')
            N+-1

        ax.plot(range(maxx), range(maxx), linewidth=3, color='k')
        plt.xlabel('y_true')
        plt.ylabel('y_pred')
        plt.title(f'Results, {dep_var}')
        plt.legend()
        plt.savefig(f'res{didx}.png')
        plt.show()

        

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
    
    
    
# get_slice(np.linspace(0,100,1000), 
#           n_slices=7,
#           get_slice=1,
#           get_indices=1)



def plot_ytr_ypr_timeseries(dm=None, 
                            verbose=False):
    if dm is None:
        dm = get_dm(verbose=verbose)
    res = dm
    for DEPVAR in sorted(list(set(res.dep_var))):
        for MODELNAME in sorted(list(set(res.modelname))):
            print(DEPVAR, MODELNAME)
            data = res[(res.modelname==MODELNAME) & (res.dep_var==DEPVAR)]
            fix, ax=set_fig_size(50,10)
            ytr = data.iloc[0].y_true
            ypr = data.iloc[0].y_pred
            plt.plot(range(len(ytr)), ytr, linewidth=1, label='y_true')
            plt.plot(range(len(ypr)), ypr, linewidth=1, label='y_pred')
            plt.title((' '*100).join([f'{MODELNAME}, {DEPVAR}' for i in range(3) ]))
            plt.legend(loc=2)
            plt.show()
            


def plot_ytr_ypr_timeseries_slices(dm=None, 
                                   verbose=False,
                                   n_slices=5,
                                   get_sliceno=2):
    if dm is None:
        dm = get_dm(verbose=verbose)
    res = dm
    for DEPVAR in sorted(list(set(res.dep_var))):
        for MODELNAME in sorted(list(set(res.modelname))):
            print(DEPVAR, MODELNAME)
            data = res[(res.modelname==MODELNAME) & (res.dep_var==DEPVAR)]
            fix, ax=set_fig_size(50,10)
            ytr = get_slice(data.iloc[0].y_true)
            ypr = get_slice(data.iloc[0].y_pred)
            plt.plot(range(len(ytr)), ytr, linewidth=1, label='y_true')
            plt.plot(range(len(ypr)), ypr, linewidth=1, label='y_pred')
            plt.title((' '*100).join([f'{MODELNAME}, {DEPVAR}' for i in range(3) ]))
            plt.legend(loc=2)
            plt.show()

# # import sys
# # from sklearn.linear_model import LinearRegression

# def make_models1(minmax_y=False,
#                 preserve_order=False):
#     '''
#     Uses local vars to produce an output dataframe.
#     References pre-defined train/test split.
#     '''
    
#     _records_dict, _records = dict(), []

#     shuffle=not(preserve_order)
    
#     if not minmax_y:
#         X_train, X_test, y_train, y_test = get_train_test_split(minmax_y=minmax_y, shuffle=shuffle)
#     else:
#         X_train, X_test, y_train, y_test, y_scaler = get_train_test_split(minmax_y=minmax_y, shuffle=shuffle)

#     outputs,model_outputs = [],dict()
#     ar = np.array
#     models = {
#         'ridge': RidgeCV, 
#         #'svm lin': LinearSVR,
#         #'svm': SVR,
#        # 'linear_reg': LinearRegression,
#         'elasticnet': ElasticNetCV
#     }
    
#     dep_vars = y_test.columns

    
#     for dep_var in ['casual', 'registered']:
#         for modelname, m in models.items():
#             label = modelname+','+dep_var
#             model = m()
         
#             if modelname == 'lasso':
#                 model = LassoCV()
#             elif modelname == 'elasticnet':
#                 model = ElasticNetCV()

#             y_true = y_test[dep_var]
#             y_tr = y_train[dep_var]

#             try:
         
#                 print(f'fitting {label}')
#                 model.fit(X_train, y_tr)
#                 y_pred = model.predict(X_test)

#                 ys = y_true, y_pred
                
#                 e1,e2,e3,e4 = mean_absolute_error(*ys),mean_absolute_percentage_error(*ys),mean_squared_error(*ys),np.nan#ar([_**0.5 for _ in mean_squared_error(*ys)])

#                 #e1,e2,e3,e4 = mean_absolute_error(*ys),mean_absolute_percentage_error(*ys),mean_squared_error(*ys),mean_squared_log_error(*ys)#ar([_**0.5 for _ in mean_squared_error(*ys)])

#                 # label = modelname+','+dep_var

#                 try:
#                     coefs = model.coef_
#                 except:
#                     coefs = np.nan

#                 if modelname == 'svm':
#                     coefs = np.nan


#                 this_output = {'label': label,
#                                'modelname': modelname,
#                                'model': model,
#                                'dep_var': dep_var,
#                                'strmodel': str(model),
#                                'coefs': coefs,
#                                'mean_absolute_error': float(e1),
#                                'mean_absolute_percentage_error': float(e2),
#                                'mean_squared_error': float(e3),
#                                'mean_squared_log_error': float(e4),
#                                'y_true': y_true,
#                                'y_pred': y_pred,
#                                'error': ''}

#                 # raw_model_outputs[modelname+','+dep_var] = (results)
#                 _records_dict[label]=this_output
#                 _records.append(this_output)

#                 # print(this_output)
#     #             print(*results)
#             except:
#                 print(sys.exc_info())
#                 keys_ = ['modelname',
#                          'model',
#                          'dep_var',
#                          'strmodel',
#                          'coefs',
#                          'mean_absolute_error',
#                          'mean_absolute_percentage_error',
#                          'mean_squared_error',
#                          'y_true',
#                          'y_pred',
#                          'error']
#                 this_output = { _:np.nan for _ in keys_ }
#                 this_output['error'] = str(sys.exc_info())
#                 _records_dict[label] = this_output
#                 _records.append(this_output)
                
#     dm = pd.DataFrame.from_records(_records)
#     dm['y_true']=dm.y_true.apply(lambda x: x.values)
#     dm['ytr']=dm.y_true.apply(lambda x: ','.join(list(map(str, x))))
#     dm['y_pred']=dm.y_pred.apply(lambda x: np.array(list(map(lambda y: round(y,1), x))))
#     dm['ypr']=dm.y_pred.apply(lambda x: ','.join(list(map(str, x))))

#     return dm, _records_dict, _records
    
    


# def make_results_scatter(dm=None,
#                          verbose=False):
#     if dm is None:
#         dm = get_dm(verbose=verbose)
        
#     for didx,dep_var in enumerate(['registered', 'casual']):
#         fix, ax = set_fig_size(14,9)
#         set_font_size(20)
#         N=0
#         d=dm[dm.dep_var==dep_var]
#         maxx = max(int(max(np.concatenate(d.y_true.values))),
#                    int(max(np.concatenate(d.y_pred.values))))

#         for idx, row in d.iterrows():
#             ax.scatter(row.y_true, row.y_pred, label=row.label, marker='o')
#             N+-1

#         ax.plot(range(maxx), range(maxx), linewidth=3, color='k')
#         plt.xlabel('y_true')
#         plt.ylabel('y_pred')
#         plt.title(f'Results, {dep_var}')
#         plt.legend()
#         plt.savefig(f'res{didx}.png')
#         plt.show()

# def get_slice(data, 
#               n_slices=5, 
#               get_sliceno=1,
#               debug=False,
#               indices=False):
#     '''
#     def get_slice(
#                 data: Iterable[Any], 
#                 n_slices=5, 
#                 get_sliceno=1,
#                 debug=False,
#                 indeces=False)
                
    
#     Get the Mth partition of a timeseries of values out of N slices, where (strictly) M < N.
    
#     :param data: the data (iterable) that is to be partitioned.
#     :param n_slices=5: the number of slices into which to partition the data.
#     :param get_sliceno=1: let get_sliceno=M, n_slices=N, then the function provides the Mth of N slices
#         (zero indexed! that is, the 0th slice is the slice with the lowest indes, and the N-1th slice
#         has the largest indices.
#     :param debug=False: if on, the function returns index and values
#     :param index=False: if on, the function returns the indices resulting slice, rather than
#         the values.  (Note however, debug takes precedence, so debug==index==1 is the same as debug=1)
    
#     Example: Imagine you have a 100cm meter stick. We want to get the M=2nd slice of n_slices=5:
    
#     Calling `get_slice(ruler_data, n_slices=5, get_slice=4)` partitions it into 5 slices, making ruler looks like
    
#                 edge of   |       |       |       |       |        |
#                    ruler->|   |   |   |   |   |   |   |   |   |    |<-endge of
#                          0| 10| 20| 30| 40| 50| 60| 70| 80| 90| 100|      ruler
#                           |   |   |   |   |   |   |   |   |   |    |
#                           |       |       |       |       |        |

#                                          partition_number
#                              0th     1st     2nd     3rd     4th
#                                           ^       ^
#                                           |       |
#                                           |result | 
#                                           | slice |
#                                           |       |
#                                       ##################
#                                       #   40      60   #
#                                       ##################
                                      

#    >> ruler_data = np.linspace(0,100,101)
#    >> print('ruler','\t',ruler_data)
#    >> print()
#    >> print('2nd of 5)
#    >> print('slices')
#    >> print( 
#             get_slice(ruler_data, 
#                       n_slices=5, 
#                       get_sliceno=2)
#         )
    
    
#     entire ruler	array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
#                             13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
#                             26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
#                             39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
#                             52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
#                             65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
#                             78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
#                             91,  92,  93,  94,  95,  96,  97,  98,  99, 100])
                            
#     2nd of 5 slices	  [40. 41. 42. 43. 44. 45. 46. 47. 48. 49. 50. 51. 52. 53. 54. 55. 56. 57. 58. 59.]   
    
    
#     >> get_slice(np.linspace(0,100, 101), n_slices=10, get_sliceno=1)
#     array([50., 51., 52., 53., 54., 55., 56., 57., 58., 59.])
    
#     >> import random
#     >> from numpy import array
#     >> Amn = array([ [ random.randint(-1000,1000) for i in range(10) ] for j in range(100)])
#     >> data = pd.DataFrame(columns=list('abcdefghij'), data=Amn)
#     >> get_slice(data, n_slices=6, get_sliceno=2)
    
#           a    b    c    d    e    f    g    h    i    j
#     32 -188  924 -777  298  407 -620  289  515 -893   61
#     33 -725 -574 -478 -454 -415 -325 -935  775  298  765
#     34 -659  239 -872  673  652 -808  641  169 -596  242
#     35  234  333 -533 -805  263   22  298  889  374  839
#     36  900 -950 -619 -375  359 -781   25  638  357  120
#     37  502  647 -369 -719   -9 -677 -380 -806  800  771
#     38  350  902   24  282 -651  -39  316 -730  939  702
#     39  185 -461  720  189 -887  739 -933 -981  795 -788
#     40 -585  800   27  711 -925 -505  -58  638  496 -790
#     41  678  792 -454  266 -644  731 -577  508  536 -858
#     42  913  640  645 -192 -213 -784    4  -37 -772  471
#     43   93  632  468  -70  528  972  339 -208 -725  847
#     44  827 -339 -159 -597  321 -221 -972 -929  117 -886
#     45  629 -524  337  135 -122  -73 -253 -660  -93   84
#     46 -314  190   37  606  -86  414 -679  494 -128  441
#     47  988 -137  451  658  565  276 -713 -388  851  859

#     '''
#     width = len(data)//n_slices
#     end_position = width*(get_sliceno+1)
#     start_position = end_position - width
#     return data[start_position:end_position]
    


# def plot_ytr_ypr_timeseries(dm=None, 
#                             verbose=False):
#     if dm is None:
#         dm = get_dm(verbose=verbose)
#     res = dm
#     for DEPVAR in sorted(list(set(res.dep_var))):
#         for MODELNAME in sorted(list(set(res.modelname))):
#             print(DEPVAR, MODELNAME)
#             data = res[(res.modelname==MODELNAME) & (res.dep_var==DEPVAR)]
#             fix, ax=set_fig_size(50,10)
#             ytr = data.iloc[0].y_true
#             ypr = data.iloc[0].y_pred
#             plt.plot(range(len(ytr)), ytr, linewidth=1, label='y_true')
#             plt.plot(range(len(ypr)), ypr, linewidth=1, label='y_pred')
#             plt.title((' '*100).join([f'{MODELNAME}, {DEPVAR}' for i in range(3) ]))
#             plt.legend(loc=2)
#             plt.show()
            


# def plot_ytr_ypr_timeseries_slices(dm=None, 
#                                    verbose=False,
#                                    n_slices=5,
#                                    get_sliceno=2):
#     if dm is None:
#         dm = get_dm(verbose=verbose)
#     res = dm
#     for DEPVAR in sorted(list(set(res.dep_var))):
#         for MODELNAME in sorted(list(set(res.modelname))):
#             print(DEPVAR, MODELNAME)
#             data = res[(res.modelname==MODELNAME) & (res.dep_var==DEPVAR)]
#             fix, ax=set_fig_size(50,10)
#             ytr = get_slice(data.iloc[0].y_true)
#             ypr = get_slice(data.iloc[0].y_pred)
#             plt.plot(range(len(ytr)), ytr, linewidth=1, label='y_true')
#             plt.plot(range(len(ypr)), ypr, linewidth=1, label='y_pred')
#             plt.title((' '*100).join([f'{MODELNAME}, {DEPVAR}' for i in range(3) ]))
#             plt.legend(loc=2)
#             plt.show()
        



        
# def plot_resultant_time_series(dm=None,
#                                verbose=False):
#     if dm is None:
#         dm = get_dm(verbose=verbose)
    
#     res = dm
#     for DEPVAR in sorted(list(set(res.dep_var))):
#         for MODELNAME in sorted(list(set(res.modelname))):
#             print(DEPVAR, MODELNAME)

#             data = res[(res.modelname==MODELNAME) & (res.dep_var==DEPVAR)]
#             fix, ax=set_fig_size(50,10)
#             ytr = data.iloc[0].y_true
#             ypr = data.iloc[0].y_pred
#             plt.plot(range(len(ytr)), ytr, linewidth=1, label='y_true')
#             plt.plot(range(len(ypr)), ypr, linewidth=1, label='y_pred')
#             plt.title((' '*100).join([f'{MODELNAME}, {DEPVAR}' for i in range(3) ]))
#             plt.legend(loc=2)
#             plt.show()
#     fix
#     print('='*80)
#     print('Getting slice 3,4 of 7 slices ...')
#     print('='*80)
#     res = dm
#     # for modelname in res.modelname:
#     #     for dep_var in res.dep_var:
#     #         data = res[(res.modelname=='modelname') & (res.dep_var=='depvar')]

#     def get_middle(data, nslices=5):
#         if nslices==2:
#             low,high=0,1
#         else:
#             low=nslices//2-1
#             slicelen = len(data)//nslices
#             high=low+slicelen

#         return data[low:high]

#     for DEPVAR in sorted(list(set(res.dep_var))):
#         for MODELNAME in sorted(list(set(res.modelname))):


#     #         DEPVAR = 'casual'
#     #         MODELNAME = 'elasticnet'
#             print(DEPVAR, MODELNAME)

#             data = res[(res.modelname==MODELNAME) & (res.dep_var==DEPVAR)]
#             fix, ax=set_fig_size(50,10)
#             ytr = data.iloc[0].y_true
#             ytr_ = get_middle(ytr)#ytr[2*len(ytr)//5:3*len(ytr)//5]
#             ypr = data.iloc[0].y_pred
#             ypr_ = get_middle(ypr)
#             plt.plot(range(len(ytr_)), ytr_, linewidth=3, label='y_true')#, linewidth=3)
#             plt.plot(range(len(ypr_)), ypr_, linewidth=3, label='y_pred')#, linewidth=3)
#             plt.title((' '*100).join([f'{MODELNAME}, {DEPVAR}' for i in range(3) ]))
#             plt.legend(loc=2)
#             plt.show()




# def get_train_test_split_preserve_order(minmax_y=False):
    
#     X_TRAIN = get_x_train()
#     Y_TRAIN = get_y_train()
#     print(X_TRAIN.shape)
#     print(Y_TRAIN.shape)

#     X_TEST = get_x_test()
#     print(X_TEST.shape)

#     if minmax_y:
#         yscaler, y = get_minmax_scaled(Y_TRAIN)
#     else:
#         y_scaler, y = None, Y_TRAIN
#     X = X_TRAIN
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

#     if minmax_y:
#         return X_train, X_test, y_train, y_test, y_scaler
    
#     return X_train, X_test, y_train, y_test


# hues = ['hour','temp_cat','weekday','month']
# xs = ['workingday',
#       'weekday',
#       'temp_cat',
#       'temp',
#       #'month',
#       'holiday',
#       'temp_cat']



# plots = []


# for h in hues:
#     for x in xs:
#         for y in ['casual', 'registered']:
#             if h==x:
#                 continue
#             this = {
#                 'y':y,
#                 'x':x,
#                 'hue':h}
#             plots.append(this)
            

# for i, plot in enumerate(plots):
#     print(plot)
#  #   do_plot(**plots[i])
    
# for dep_var in ['count','registered','casual']:
#     for wd in (0,1):
#         d = dx[dx.workingday==wd]
#         fig, ax = make_big()
#         plt.scatter(range(len(d.index)), d[dep_var])
#         plt.ylabel('count')
#         plt.xlabel('Time (sample index of date sorted data)')
#         plt.title(f'workingday={wd}, dep_var={dep_var}')
#         plt.show()

# d = dx[dx.workingday==1]
# fig, ax = make_big()
# plt.scatter(range(len(d.index)), d['count'])
# plt.ylabel('count')
# plt.xlabel('Sample Index (sorted by date)')
# plt.show()
# # plt.hist(d.temp_cat)
# # plt.title('temp categories')


# fig, ax=make_big()
# sns.boxplot(y='count',x='season', hue='year', data=dx[dx.workingday==1])
# plt.show()

# # sns.boxplot(y='count',x='workingday',data=dd)
# # plt.show()


# from scipy.constants import convert_temperature

# def get_datetime(x):
#     return datetime.datetime.fromisoformat(x)


# season_mapper = {
#     1: 'spring',
#     2: 'summer',
#     3: 'fall',
#     4: 'winter'
# }

# weekday_mapper = {
#     0: 'monday',
#     1: 'tuesday',
#     2: 'wednesday',
#     3: 'thurday',
#     4: 'friday',
#     5: 'saturday',
#     6: 'sunday'
    
# }


    
# def temp_mapper(temp=None, 
#                 low=12, 
#                 high=29):

#     if temp is None:
#         raise ValueError(f'temp must be a number. got {temp}')

#     if temp<low//2:
#         return 'freezing'
#     elif temp<low:
#         return 'cold'
#     elif temp>=high*1.33:
#         return 'hades'
#     elif temp <= high:
#         return 'nice'
#     else:
#         return 'hot'



# do_little_eda()

# # df = process_input(infile='training_data.csv')
# df = get_df()
# dts = df['datetime'].copy()
# df['hour'] = dts.apply(func= lambda x: get_datetime(x).hour)
# df['weekday'] = dts.apply(func= lambda x: get_datetime(x).weekday())
# df['weekday'] = df.weekday.apply(func=lambda x: weekday_mapper[x])
# df['year'] = dts.apply(func= lambda x: (get_datetime(x).year - 2011)%2011+1)
# df['month'] = dts.apply(func=lambda x: get_datetime(x).month )
# # df['temp_C'] = df.temp
# # df['temp_F'] = df.temp.apply(func=lambda x: convert_temperature(x, 'c', 'f'))
# df['approx_temp'] = df.temp.apply(func=lambda x: round(x,1))

# # temperature categories
# low_temp_F, high_temp_F=49,78


# df['temp_cat'] = df.temp.apply(func=lambda x : temp_mapper(temp=x,
#                                                               low=convert_temperature(low_temp_F,'f','c'),
#                                                               high=convert_temperature(high_temp_F,'f','c'))
#                                  )
# df['season'] = df.season.apply(func=lambda x: season_mapper[x])



# def format_title(**kwargs):
#     return ';'.join([ f'{k}={v}' for k,v in kwargs.items() ])
    
# def do_plot(x,y,hue):
#     dx=df.copy()
#     make_big()
#     sns.boxplot(x=dx[x],y=dx[y],hue=dx[hue])
#     plt.xticks(rotation=70)
#     plt.title(format_title(y=y, x=x, hue=hue))
#     plt.legend(loc=4)
#     plt.show()

    

# hues = ['hour','temp_cat', 'weekday']
# xs = ['workingday','weekday','temp_cat','month','holiday','temp_cat']



# plots1 = []


# for h in hues:
#     for x in xs:
#         for y in ['casual', 'registered']:
#             if h==x:
#                 continue
#             this = {
#                 'y':y,
#                 'x':x,
#                 'hue':h}
#             plots1.append(this)
    
# if SHOW_ALL_PLOTS:
#     print(f'Plotting...')
#     pprint(plots1)
#     [ do_plot(**plots1[i]) for i in range(len(plots1)) ]
    

# dx['hr_season'] = [ str(get_dt(dx.iloc[idx].datetime).time().hour) + ','+str(dx.iloc[idx].season) for idx in range(len(dx))]
# dx['yr_month'] = [ str(get_dt(x).year) + ','+str(get_dt(x).month) for x in dx.datetime ]
# dx['yr_season'] = [ str(get_dt(dx.iloc[idx].datetime).year) + ','+str(dx.iloc[idx].season) for idx in range(len(dx)) ]

# dx['weekday'] = [ weekday_mapper[datetime.datetime.fromisoformat(dx.iloc[idx].datetime).weekday()] for idx in range(len(dx)) ]
# dx['yr_season_weekday']= [ row.yr_season+','+str(row.weekday) for idx, row in dx.iterrows() ] 
# dx['season_weekday'] = [ ','.join(x.split(',')[1:]) for x in dx.yr_season_weekday ]

# dx['hr_weekday'] = [ ','.join(list(map(str,[row.hour,row.weekday]))) for idx, row in dx.iterrows() ]
# dx['yr_season_weekday_hr'] = [ row.yr_season_weekday+','+str(row.hour) for idx,row in dx.iterrows() ]

# dx['season_weekday_hr'] =  [ ','.join(x.split(',')[1:]) for x in dx['yr_season_weekday_hr'] ]
# dx['workday_hr'] = [ ','.join([str(row.workingday), str(row.hour)]) for idx, row in dx.iterrows() ]

# print(dx[['datetime','year','hour','month','hr_season','yr_month','yr_season', 'weekday']])
# print()
# print('feature name,','number of disctint values')
# print('set of distinct values')
# print('='*80)
# for time_var in [#'datetime',
#                  'season',
#                  'year',
#                  'hour',
#                  'month',
#                  'hr_season',
#                  'yr_month',
#                  'yr_season',
#                  'weekday']:
#     print()
#     print(time_var,len(list(sorted(list(set(dx[time_var]))))))
#     print(list(sorted(list(set(dx[time_var])))))
# # dx



# dx = dx.reset_index(drop=True)

# y_vars = [
#         'casual', 
#         'registered', 
#         'count']

# x_continuous = [
#         'temp',
#         'humidity',
#         'windspeed']

# drop_timevars = [
#         'datetime',
#         'index',
#         'season',
#         'year',
#         'hour',
#         'month']

# x_categorical = [ col for col in dx.columns if col not in y_vars+x_continuous+drop_timevars ] 

# print('Y:', y_vars)
# print('X_cont:', x_continuous)
# print('X_cat:', x_categorical)
# print('All:', dx.columns)

# d_y = dx[y_vars]
# d_xcat = dx[x_categorical]
# d_xcont = dx[x_continuous]



# x_cont_scaler = MinMaxScaler()
# x_cont_scaled = x_cont_scaler.fit_transform(d_xcont.to_numpy())
# d_xcont_scaled = pd.DataFrame(columns=d_xcont.columns,
#                               data=x_cont_scaled)

# d_xcont_scaled
# check = pd.DataFrame(columns=d_xcont_scaled.columns,
#              data=x_cont_scaler.inverse_transform(d_xcont_scaled.to_numpy()))

# # y_cont_scaler = MinMaxScaler()
# # y_cont_scaled = x_cont_scaler.fit_transform(d_y.to_numpy())
# # d_ycont_scaled = pd.DataFrame(columns=d_y.columns,
# #                               data=y_cont_scaled)

# d_ycont_scaled = d_y

# d_ycont_scaled

# X_train, X_test, y_train, y_test = get_train_test_split()