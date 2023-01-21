from copy import deepcopy
from dataclasses import dataclass
import numpy as np



class FigureUtilities:
    
    def set_fig_size(width=18,height=10):
        fig,ax=plt.subplots(1,1)
        fig.set_size_inches(width,height)
        
    def set_font_size(size=18):
        font = {'family' : 'verdana',
                'size'   : size}
        rc('font', **font)
        
    def make_big(f=18,w=18,h=10):
        set_font_size(size=f)
        set_fig_size(width=w,height=h)
 




@dataclass
class ColumnData:
    
    X_COLS = ['season', 
              'holiday', 
              'workingday', 
              'weather', 
              'temp', 
              'humidity',
              'windspeed', 
              'datetime', 
              'tempcat', 
              'hour', 
              'weekday', 
              'year', 
              'month',
              'temp_cat']
    
    Y_COLS= ['casual','registered']
    
    CONT = {'x': ['humidity', 'temp', 'windspeed'], 
                  'y': ['casual', 'registered']}
    CAT =  {'x': [
                      'season',
                      'holiday',
                      'workingday',
                      'weather',
                      'datetime',
                      'tempcat',
                      'hour',
                      'weekday',
                      'year',
                      'month',
                      'temp_cat'],
                    'y': []}
    
    
    def __init__(self):
        self.x_cols = deepcopy(ColumnData.X_COLS)
        self.y_cols = deepcopy(ColumnData.Y_COLS)
        self.continuous = deepcopy(ColumnData.CONT)
        self.categorical = deepcopy(ColumnData.CAT)

        
    def get_continuous(self, 
                       xy = 'x',
                       add_cols=[], 
                       del_cols=[]):
        '''
        def get_continuous(self, 
                   xy = 'x',
                   add_cols=[], 
                   del_cols=[]):
        
        Modifies returns the updated continuous variables given we want to add some and delete some.
        
        :param xy: {'x', 'y'}:
            Specifies whether to get the x or y continuous. Defaults to getting x.  Can get the y_continuous also.
        :param add_cols: List[str]
            Specifies columns to add to the existing properies of this instance.
        :param del_cols: List[str]
            Specifies columns to remove from the existing properties of this instance.        
        '''
        
        cont = list(
                sorted(
                    list(
                        set(self.continuous[xy]).union(set(add_cols)) - set(del_cols)
                        )
                    )
                )
        
        # cont = list(sorted(list(set(cont))))
        return cont
    
    def set_continuous(self, xy='x', add_cols=[], del_cols=[]):
        '''
        def set_continuous(self, xy_='x', add_cols_=[], del_cols_=[])
        
        Sets the property self.continuous[xy]. Recall xy in {'x','y'} defaults to 'x',
        then returns the value set.
        '''
        
        self.continuous[xy] = self.get_continuous(xy=xy,
                                                   add_cols=add_cols,
                                                   del_cols=del_cols)
        
        self.x_cols = [ _ for _ in self.x_cols if _ in set(self.continuous['x']).union(set(self.categorical['x'])) ]
        self.y_cols = [ _ for _ in self.x_cols if _ in set(self.continuous['y']).union(set(self.categorical['y'])) ]
        
        return self.continuous
        
    
    def get_categorical(self,
                        xy='x',
                        add_cols=[],
                        del_cols=[]):
        '''
        def get_categorical(self, 
                   xy = 'x',
                   add_cols=[], 
                   del_cols=[]):
        
        Modifies returns the updated categorical variables given we want to add some and delete some.
        See ColumnsData.get_continuous?
        
        :param xy: {'x', 'y'}:
            Specifies whether to get the x or y continuous. Defaults to getting x.  Can get the y_continuous also.
        :param add_cols: List[str]
            Specifies columns to add to the existing properies of this instance.
        :param del_cols: List[str]
            Specifies columns to remove from the existing properties of this instance.        
        '''
    
        cat = list(set(self.categorical[xy]).union(set(add_cols)) - set(del_cols))
        cat = list(sorted(list(set(cat))))
        return cat
    
    
    def set_categorical(self,
                        xy='x',
                        add_cols=[],
                        del_cols=[]):
        
        '''
        def set_categrical(self, xy_='x', add_cols_=[], del_cols_=[])
        
        Sets the property self.categorical[xy]. Recall xy in {'x','y'} defaults to 'x',
        then returns the value set.
        '''
          
        self.categorical[xy] = self.get_categorical(xy=xy,
                                                   add_cols=add_cols,
                                                   del_cols=del_cols)
        
        self.x_cols = [ _ for _ in self.x_cols if _ in set(self.continuous['x']).union(set(self.categorical['x'])) ]
        self.y_cols = [ _ for _ in self.x_cols if _ in set(self.continuous['y']).union(set(self.categorical['y'])) ]
        
        return self.categorical
        
    
                      
        
# @dataclass
# class BoxplotFactory(object):

#     def format_title(**kwargs):
#         return ';'.join([ f'{k}={v}' for k,v in kwargs.items() ])

#     hues = ['hour','temp_cat','weekday','month']
#     xs = ['workingday',
#           'weekday',
#           'temp_cat',
#           'temp',
#           #'month',
#           'holiday',
#           'temp_cat']
#     yy = ['casual',
#           'registered']
    
#     def __init__(self,
#                  hues=None,
#                  xs=None,
#                  yy=None):
#         self.hues = hues or BoxplotFactory.hues
#         self.xs = xs or BoxplotFactory.xs
#         self.yy = yy or BoxplotFactory.yy
#         self.plots = self.get_plots()
#         self.data = data or BoxplotFactory.data
        
    
#     def do_plot(x,y,hue,data=None):
  
#         if data is None:
#             # raise ValueError(f'Expected data:pd.DataFrame; got {data}:{type(data)}')
#             print(f'Expected data to be specified; got {data}:{type(data)}')
#             print(f'setting data=get_dx()... getting...')
#             data = get_dx()
            
#         dx_=data.copy()
#         make_big()
#         sns.boxplot(x=dx_[x],y=dx_[y],hue=dx_[hue])
#         plt.xticks(rotation=70)
#         plt.title(format_title(y=y, x=x, hue=hue))
#         plt.legend(loc=4)
#         plt.show()
    
#     def get_plots(self, verbose=False):
#         plots = []
#         for h in self.hues:
#             for x in self.xs:
#                 for y in self.yy:
#                     if h==x:
#                         continue
#                     this = {
#                         'y':y,
#                         'x':x,
#                         'hue':h}
#                     verbose and print(this)
#                     plots.append(this)
#         return plots

            
#     def plot_all(self, verbose=False):
#         for i, plot in enumerate(self.plots):
#             verbose and print(plot)
#             self.do_plot(**self.plots[i])
    
#     def set_plots(self):
#         self.plots = self.get_plots()

        
        

@dataclass
class Mappers:
    def get_dt(dt):
        return datetime.datetime.fromisoformat(dt)
    
    _season_mapper = {
        1: 'spring',
        2: 'summer',
        3: 'fall',
        4: 'winter'
    }
    
     _weekday_mapper = {
        0: 'monday',
        1: 'tuesday',
        2: 'wednesday',
        3: 'thurday',
        4: 'friday',
        5: 'saturday',
        6: 'sunday'

    }
    def season_mapper(s_ord,
                       asdict=False):
        return Mappers._season_mapper[s_ord]
    
    
    def weekday_mapper(w_ord, asdict=False):
        # if asdict:
        #     return Mappers.weekday_mapper
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