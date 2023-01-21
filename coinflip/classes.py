from copy import deepcopy
from dataclasses import dataclass
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import PolynomialFeatures, Normalizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.svm import SVR, LinearSVR

class FigureUtilities:
    '''
    Figure generation helpers.
    '''
    
    def set_fig_size(width=18,height=10):
        fig,ax=plt.subplots(1,1)
        fig.set_size_inches(width,height)
        return fig ,ax

    def set_font_size(size=18):
        font = {'family':'verdana', 'size':size}
        rc('font', **font)

    def make_big(f=18,w=18,h=10):
        set_font_size(size=f)
        fig, ax = set_fig_size(width=w,height=h)
        return fig, ax
    
    def format_title(**kwargs):
        return ';'.join([ f'{k}={v}' for k,v in kwargs.items() ])
    

@dataclass
class Mappers:
    '''
    Functions and objects that transform data.
    '''
    
    _season_mapper = {
        1: 'spring',
        2: 'summer',
        3: 'fall',
        4: 'winter'}
    
    _weekday_mapper = {
        0: 'monday',
        1: 'tuesday',
        2: 'wednesday',
        3: 'thurday',
        4: 'friday',
        5: 'saturday',
        6: 'sunday'}
    
    _model_mapper = {
        'ridge': RidgeCV,
        # 'elasticnet': ElasticNetCV,
        'lin svm': LinearSVR}
    
    _encoder_mapper = {'onehot' : OneHotEncoder,
                       'ordinal': OrdinalEncoder}
    
    _scaler_mapper = {'minmax': MinMaxScaler,
                      'robust': RobustScaler,
                      'standard': StandardScaler,
                      'maxabs': MaxAbsScaler}
    
    
    @classmethod
    def get_mappers(cls):
        mappers =  {'season': cls._season_mapper,
                    'weekday': cls._weekday_mapper,
                    'model': cls._model_mapper,
                    'encoder': cls._encoder_mapper,
                    'scaler': cls._scaler_mapper}
        return mappers
    
        
    def __init__(self):
        self.mappers = Mappers.get_mappers()
        

        
    def get_dt(dt):
        '''
        Returns a datetime object from a string.
        '''
        return datetime.datetime.fromisoformat(dt)
    
    
    def encoder_mapper(encoder_key):
        '''
        Returns an encoder.
        
         Mappers._encoder_mapper = {'onehot', : OneHotEncoder,
                            'ordinal': OrdinalEncoder}             
        
        '''
    
        if encoder_key not in Mappers._encoder_mapper.keys():
            raise ValueError(f'Argument must be in {list(Mappers._encoder_mapper.keys())}; got {encoder_key}')
        return Mappers._encoder_mapper[encoder_key]
    
    
    def scaler_mapper(scaler_key):
        '''
        Returns a scaler.
        
        Mappers._scaler_mapper = {'minmax': MinMaxScaler,
                                  'robust': RobustScaler,
                                  'standard': StandardScaler,
                                  'maxabs': MaxAbsScaler}
        '''
        
        if scaler_key not in Mappers._scaler_mapper.keys():
            raise ValueError(f'Argument must be in {list(Mappers._scaler_mapper.keys())}; got {scaler_key}')
        return Mappers._scaler_mapper[scaler_key]
    
        
    def season_mapper(s_ord):
        '''
        Maps ordinal season to spring, summer, fall, winter.
        
         _season_mapper = {
        1: 'spring',
        2: 'summer',
        3: 'fall',
        4: 'winter'}
        
        '''
        if s_ord not in Mappers._season_mapper.keys():
            raise ValueError(f'Argument must be in {list(Mappers._season_mapper.keys())}; got {s_ord}.')
        return Mappers._season_mapper[s_ord]
    
    def weekday_mapper(w_ord):
        '''
        Maps ordinal day of the week to Monday, Tuesday, ..., Sunday.
        
         _weekday_mapper = {
        0: 'monday',
        1: 'tuesday',
        2: 'wednesday',
        3: 'thurday',
        4: 'friday',
        5: 'saturday',
        6: 'sunday'}
        
        '''
        if w_ord not in Mappers._weekday_mapper.keys():
            raise ValueError(f'Argument must be in {list(Mappers._weekday_mapper.keys())}; got {s_ord}.')
        return Mappers._weekday_mapper[w_ord]
    
    
    def model_mapper(label):
        '''
        Maps modelname to a model class.
        
        Mappers._model_mapper = {
        'ridge': RidgeCV,
        # 'elasticnet': ElasticNetCV,
        'lin svm': LinearSVR}
        '''
        if label not in Mappers._model_mapper.keys():
            raise ValueError(f'Argument must be in {list(Mappers._model_mapper.keys())}; got {label}.')
        return Mappers._model_mapper[label]
        
    def temp_mapper(temp=None, 
                    low=12, 
                    high=29):
        '''
        Mappers.temp_mapper(temp)->{'hades','hot','nice','cold','freezing'}
        
        the boundary values are:
        '''
        if temp is None:
            raise ValueError(f'Function agurment temp must be a number, got {temp}.')
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
        
        @classmethod
        def temps(cls):
            return cls.temp_mapper
        @classmethod
        def meekdays(cls):
            return cls.model_mapper
        @classmethod
        def seasons(cls):
            return cls.season_mapper
        @classmethod
        def scalers(cls):
            return cls.scaler_mapper
        @classmethod
        def encoders(cls):
            return cls.encoder_mapper