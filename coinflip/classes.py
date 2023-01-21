from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import datetime


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
    
    def get_dt(dt):
        '''
        Returns a datetime object from a string.
        '''
        return datetime.datetime.fromisoformat(dt)
    
    def season_mapper(s_ord):
        return Mappers._season_mapper[s_ord]
    
    def weekday_mapper(w_ord):
        return Mappers._weekday_mapper[w_ord]
    
    def temp_mapper(temp=None, 
                    low=12, 
                    high=29):
        '''
        functon takes a temp in celisius and outputs a category
        
       def temp_mapper_(temp=None, 
                low=12, 
                high=29):
        
        Mappers.temp_mapper(temp)->{'hades','hot','nice','cold','freezing'}
        
        the boundary values are:
        '''
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