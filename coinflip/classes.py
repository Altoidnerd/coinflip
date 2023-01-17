from dataclasses import dataclass
import numpy as np
from coinflip.modeling import get_dx, get_plots, do_plot


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
class BoxplotFactory(object):

    def format_title(**kwargs):
        return ';'.join([ f'{k}={v}' for k,v in kwargs.items() ])
    



    hues = ['hour','temp_cat','weekday','month']
    xs = ['workingday',
          'weekday',
          'temp_cat',
          'temp',
          #'month',
          'holiday',
          'temp_cat']
    yy = ['casual',
          'registered']
    
        
    def __init__(self,
                 hues=None,
                 xs=None,
                 yy=None):
        self.hues = hues or BoxplotFactory.hues
        self.xs = xs or BoxplotFactory.xs
        self.yy = yy or BoxplotFactory.yy
        self.plots = self.get_plots()
        self.data = data or BoxplotFactory.data
        
    
    def do_plot(x,y,hue,data=None):
  
        if data is None:
            # raise ValueError(f'Expected data:pd.DataFrame; got {data}:{type(data)}')
            print(f'Expected data to be specified; got {data}:{type(data)}')
            print(f'setting data=get_dx()... getting...')
            data = get_dx()
            
        dx_=data.copy()
        make_big()
        sns.boxplot(x=dx_[x],y=dx_[y],hue=dx_[hue])
        plt.xticks(rotation=70)
        plt.title(format_title(y=y, x=x, hue=hue))
        plt.legend(loc=4)
        plt.show()
    
    def get_plots(self, verbose=False):
        plots = []
        for h in self.hues:
            for x in self.xs:
                for y in self.yy:
                    if h==x:
                        continue
                    this = {
                        'y':y,
                        'x':x,
                        'hue':h}
                    verbose and print(this)
                    plots.append(this)
        return plots

            
    def plot_all(self, verbose=False):
        for i, plot in enumerate(self.plots):
            verbose and print(plot)
            self.do_plot(**self.plots[i])
    
    def set_plots(self):
        self.plots = self.get_plots()

        
        

@dataclass
class Mappers:
    def get_dt_(dt):
        return datetime.datetime.fromisoformat(dt)
    season_mapper = {
        1: 'spring',
        2: 'summer',
        3: 'fall',
        4: 'winter'
    }
    def season_mapper_(s_ord,
                       asdict=False):
        # if asdict:
        #     return Mappers.season_mapper
        return Mappers.season_mapper[s_ord]
    
    weekday_mapper = {
        0: 'monday',
        1: 'tuesday',
        2: 'wednesday',
        3: 'thurday',
        4: 'friday',
        5: 'saturday',
        6: 'sunday'

    }
    
    def weekday_mapper_(w_ord, asdict=False):
        # if asdict:
        #     return Mappers.weekday_mapper
        return Mappers.weekday_mapper[w_ord]
    
    def temp_mapper_(temp=None, 
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
        
        
    def get_mappers(asdict=False):
        '''
        Usage: get_dt, smap, wmap, tmap = Mappers.get_mappers(asdict=True)
        
        :param adsict: bool
            If asdict, then smap and wmap will be returned as dictionaries.  Otherwise, they'll be functions.
        
        '''
        gdt = Mappers.get_dt_
        tmap = Mappers.temp_mapper_
        smap = Mappers.season_mapper_
        wmap = Mappers.weekday_mapper_
        
        if asdict:
            smap = Mappers.season_mapper
            wmap = Mappers.weekday_mapper
        # return Mappers.get_dt_, Mappers.season_mapper_, Mappers.weekday_mapper_, Mappers.temp_mapper_
        return gdt, smap, wmap, tmap
    
    def get_temp_boundaries(mint=0,maxt=46,N=1000):
        temps = np.linspace(mint,maxt,N)
        tmapper = Mappers.temp_mapper_
        
        i=0
        prev_cat = 'freezing'
        this_temp=temps[i]
        if i>-0:
            prev_temp=temps[i-1]
        this_cat=tmapper(this_temp)
        if not this_cat == prev_cat:
            print('prev_temp',prev_temp,'prev_cat',prev_cat)
            print('this_temp',this_temp,'this_cat',this_cat)