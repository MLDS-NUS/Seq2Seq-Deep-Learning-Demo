import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from ipywidgets import widgets
from ipywidgets import VBox, HBox, HTML
import numpy as np
import re
from pathlib import Path
import ast
from ml_collections import FrozenConfigDict
from lib.lfgenerator import LorenzRandFGenerator
from lib.seq2seq_model import RNNTextGeneration, RNNWordGeneration
from lib.lfgenerator import ShiftGenerator

CONFIG = FrozenConfigDict({'shift': dict(LENGTH = 100,
                                        NUM = 20,
                                        SHIFT = 30),
                            'convo': dict(LENGTH = 100,
                                        NUM = 20,
                                        FILTER = [0.002, 0.022, 0.097, 0.159, 0.097, 0.022, 0.002]),
                            'lorenz': dict(NUM = 10, 
                                            K=1, J=10, 
                                            LENGTH=128 )})
class DataPlotter:

    """
    A class to create an interactive plot of sequential input, output pair. Have a button to randomly refresh different set of input, output pair.
    """

    def __init__(self,input, output, num_of_plots=2,title='', descrip='', subplot_title=()) -> None:

        """
        Args:
            input (_type_): The input have shape (num_of_data, length).
            output (_type_): The output have shape (num_of_data, length).
            num_of_plots (int, optional): The number of plots, if only have one then put input,output on same plot, otherwise put them side by side. Defaults to 2.
            title (str, optional): Title of the plot.
            descrip (str, optional): Description of the plot. 
            subplot_title (tuple, optional): A tuple consits of name of the subplots.
        """

        self.num_of_plots = num_of_plots
        self.input = input
        self.output = output
        self.descrip = descrip
        self.title = title
        self.subplot_title = subplot_title
        self.debug = widgets.Output() 
        if num_of_plots not in (1,2):
            raise Exception('Can only have 1 or 2 plots')
        self.trace1_name = 'Input'
        self.trace2_name = 'Output'

    def plot(self):
        button_layout = widgets.Layout(
                        width='20%',
                        )
        text_layout = widgets.Layout(display='flex',
                flex_flow='column',
                align_items='center',
                width='60%')
        emptybox_layout = widgets.Layout(display='flex',
                        flex_flow='column',
                        align_items='center',
                        width='20%')              
        descrip = HTML(f' <font size="+1"><b>{self.title}</b> {self.descrip} </font>')
        button = widgets.Button(description="Refresh")
        
        button_box = widgets.HBox([HTML('&nbsp;'*5), button],layout=button_layout)
        descrip_box = widgets.HBox([descrip], layout=text_layout)
        empty_box = widgets.HBox([HTML()], layout=emptybox_layout)
        # Calculate range of plot.
        plot_range_in = np.abs(np.array(self.input)).max()+0.2
        plot_range_out = np.abs(np.array(self.output)).max()+0.2
          
        # Setup subplots, if only have 1 plots then input and output are together on col1.
        self.fig = go.FigureWidget(make_subplots(rows=1, cols=self.num_of_plots, shared_yaxes=False,subplot_titles=self.subplot_title))
        
        # Randomly choose one input,output pair to plot.
        index = random.randint(0,len(self.input)-1)
        input, output = self.input[index], self.output[index] 

        self.fig.add_trace(
            go.Scatter(y=input, name=self.trace1_name),
            row=1, col=1
        )
        self.fig.add_trace(
            go.Scatter(y=output, name=self.trace2_name),
            row=1, col=self.num_of_plots
        )  

        # Do not show lengend if have two seperate plots
        self.fig.update_layout(showlegend=False)
        
        if self.num_of_plots == 1:
            self.fig.update_layout(showlegend=True)
            self.fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font_size=17
            ))
        self.fig.update_layout(font_size=15)
        self.fig.update_annotations(font_size=18)
        self.fig.update_yaxes(range=[-plot_range_out, plot_range_out], row=1,col=self.num_of_plots)
        self.fig.update_yaxes(range=[-plot_range_in, plot_range_in], row=1,col=1)
        self.fig.update_layout(margin=dict(l=0,t=35),)
        

        # Click event for button
        
        button.on_click(self.response)
        container = HBox([button_box, descrip_box,empty_box])
        return VBox([container,self.fig])

    def response(self, b):     
        index = random.randint(0,len(self.input)-1)
        with self.fig.batch_update():
            self.fig.data[0].y = self.input[index]
            self.fig.data[1].y = self.output[index]

class ShiftPlotter(DataPlotter):
    def __init__(self, k=30):
        self.shift = k
        input, output = self._generate_data()
        super().__init__(input=input, output=output, num_of_plots=2, 
                        title='Shift Seqeunce:',
                        descrip=f'Shift the input to the right by {self.shift} timesteps.',
                        subplot_title=('Input Sequence', 'Output Sequence'), )
    
    def _generate_gaussian(self, seq_length):
        def rbf_kernel(x1, x2, variance = 1):
            from math import exp
            return exp(-1 * ((x1-x2) ** 2) / (2*variance))
        def gram_matrix(xs):
            return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]
        xs = np.arange(seq_length)*0.1
        mean = [0 for _ in xs]
        gram = gram_matrix(xs)
        ys = np.random.multivariate_normal(mean, gram)
        return ys
    
    def plot(self):
        vbox = super().plot()
        fig = vbox.children[1]
        range = fig['layout'].yaxis.range[1]+0.2
        x = CONFIG.shift.LENGTH-self.shift
        fig.add_shape(
            type='rect', xref='x', yref='y',
            x0=0, x1=x, y0=-range+1, y1=range-1, fillcolor="LightSkyBlue",opacity=0.3, line_color="LightSkyBlue"
        , row=1,col=1)
        fig.add_shape(
            type='rect', xref='x', yref='y',
            x0=self.shift, x1=CONFIG.shift.LENGTH, y0=-range+1, y1=range-1, fillcolor="LightSkyBlue",opacity=0.3, line_color="LightSkyBlue"
        , row=1,col=2)
        return vbox

    def _generate_data(self):
        input = []
        output = []
        for _ in range(CONFIG.shift.NUM):
            data = self._generate_gaussian(CONFIG.shift.LENGTH)
            input.append(data)
            output.append(np.concatenate((np.zeros(self.shift), data[:-self.shift])))
        return input, output

class LorenzPlotter(DataPlotter):
    def __init__(self):
        input, output = self._generate_data()
        super().__init__(input=input, output=output, num_of_plots=1, 
                        title=f'',
                        descrip=f'The output is the response of input defined by the Lorenz96 system.',
                        subplot_title=('Intput/Output Sequences',''), )
    
    def _generate_data(self):
        lorenz_generator = LorenzRandFGenerator({'n_init':CONFIG.lorenz.NUM, 
                                                    'K':CONFIG.lorenz.K, 
                                                    'J':CONFIG.lorenz.J,
                                                    'path_len':CONFIG.lorenz.LENGTH})

        input, output = lorenz_generator.generate(scale=False)

        return input.squeeze(-1)[:,1:], output.squeeze(-1)[:,1:]


class ConvoPlotter(DataPlotter):
    def __init__(self):
        self.filter = CONFIG.convo.FILTER
        input, output = self._generate_data()
        super().__init__(input=input, output=output, num_of_plots=2, 
                        title='Convolution of sequence with a filter',
                        descrip=f'',
                        subplot_title=('Input Sequence', 'Output Sequence'), )
    
    def _generate_gaussian(self, seq_length):
        def rbf_kernel(x1, x2, variance = 1):
            from math import exp
            return exp(-1 * ((x1-x2) ** 2) / (2*variance))
        def gram_matrix(xs):
            return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]
        xs = np.arange(seq_length)*0.1
        mean = [0 for _ in xs]
        gram = gram_matrix(xs)
        ys = np.random.multivariate_normal(mean, gram)
        return ys
    
    def plot(self):
        vbox = super().plot()
        hbox = vbox.children[0]
        range = self.fig['layout'].yaxis.range[1]+0.2
        self.filter_box = widgets.Text(
                        value=str(CONFIG.convo.FILTER)[1:-1],
                        placeholder='Enter a filter',
                        description='Filter:',
                        disabled=False,
                        layout = widgets.Layout(width='400px')
                    )
        button =  hbox.children[0].children[1]
        descrip = hbox.children[1].children[0]

        button_layout = widgets.Layout(
                        width='35%',
                        )
        text_layout = widgets.Layout(display='flex',
                flex_flow='column',
                align_items='center',
                width='30%')
        emptybox_layout = widgets.Layout(display='flex',
                        flex_flow='column',
                        align_items='center',
                        width='35%')      

        button_box = widgets.HBox([self.filter_box, button],layout=button_layout)
        descrip_box = widgets.HBox([descrip], layout=text_layout)
        empty_box = widgets.HBox([HTML()], layout=emptybox_layout)

        hbox = HBox([button_box,descrip_box, empty_box])
        

        new_vbox = VBox([hbox, self.fig,self.debug])
        return new_vbox

    def _generate_data(self):
        input = []
        output = []
        for _ in range(CONFIG.convo.NUM):
            data = self._generate_gaussian(CONFIG.convo.LENGTH)
            data = 2*np.random.random(CONFIG.convo.LENGTH)-1
            input.append(data)
            output.append(np.convolve(data, self.filter, mode='same'))
        return input, output

    def response(self, b):
        raw_filter = re.sub(r"[^0-9.-]+", ", ", self.filter_box.value) 

        
        #If filter changed then update data
        if list(self.filter) != ast.literal_eval('[' + raw_filter + ']'):
            self.filter = ast.literal_eval('[' + raw_filter + ']')
            self.filter_box.value = raw_filter
            self.input, self.output = self._generate_data()

        index = random.randint(0,len(self.input)-1)
        with self.fig.batch_update():
            self.fig.data[0].y = self.input[index]
            self.fig.data[1].y = self.output[index]
        
        plot_range_in = np.abs(np.array(self.input)).max()+0.2
        plot_range_out = np.abs(np.array(self.output)).max()+0.2
        self.fig.update_yaxes(range=[-plot_range_out, plot_range_out], row=1,col=self.num_of_plots)
        self.fig.update_yaxes(range=[-plot_range_in, plot_range_in], row=1,col=1)
       


class TextGenerator:
    

    def __init__(self) -> None:
        self.wordmodel = RNNWordGeneration.load_from_checkpoint('resources/saved_models/text/wordgeneration_demo.ckpt', load_data=False)
        self.charmodel = RNNTextGeneration.load_from_checkpoint('resources/saved_models/text/text_generation_demo.ckpt', load_data=False)
        
        self.model = self.charmodel
        self.length = 600

        self.debug = widgets.Output() 

    def plot(self):
        t0 = HTML(f' <font size="+0.4">Model Type: </font>')
        button = widgets.ToggleButtons(
            # options=['Character', 'Word'],
            options=['Character'],
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
        )
        def update_output():
            output = self.model.predict(input_box.value, length=self.length)
            output = output.replace('\n', '<br>')
            # output = '. '.join(map(lambda s: s.strip().capitalize(), output.split('.')))
            output_box.value = f' <font size="+0.4">{output} </font>'

        def button_changed(change):
            if change['new'] == 'Word':
                self.length = 100
                self.model = self.wordmodel
            else:
                self.length = 600
                self.model = self.charmodel
            update_output()

        button.observe(button_changed, names='value')


        t1 = HTML(f' <font size="+0.4">Input text: </font>')
        input_box = widgets.Text(
                                placeholder='Type something',
                                disabled=False,
                                layout=widgets.Layout(width='500px', height='300px')
                            )
        t2 = HTML(f' <font size="+0.4">Output text: </font>')
        output_box = widgets.HTML(
                        placeholder='Waiting for input',
                        layout=widgets.Layout(width='500px', height='800px')
                    )
        input_box.value = 'This is a very good'

        update_output()
        


        def on_value_change(change):
            update_output()
                
        input_box.observe(on_value_change, names='value')
        output = HBox([t1, input_box,t2, output_box])
        button_box = HBox([t0 ,button])
        return VBox([button_box, output, self.debug])


class ModelEvaluation(DataPlotter):

    def __init__(self, model, path, title='', descrip='') -> None:
        
        input, output = self._generate_data()

        super().__init__(input, output, num_of_plots=1, title=title, descrip=descrip)

    def _generate_data(self):
        raise NotImplementedError


class LorenzEvaluation(ModelEvaluation):

    def __init__(self, model, path, title='Lorenz System') -> None:
        self.model = model
        self.path = path
        K, J, length= map(lambda x:int(x), Path(self.path).stem.split('_')[1:])
        descrip = f'K = {K}, J = {J}, Length = {length}.'
        self.lorenz_generator = LorenzRandFGenerator({'data_num': 20 ,'n_init':20, 
                                                    'K':K, 
                                                    'J':J,
                                                    'path_len':length})

        super().__init__(model, path, title=title, descrip=descrip)

        self.trace1_name = 'Output'
        self.trace2_name = 'Prediction'

    def _generate_data(self):
            
        input, output = self.lorenz_generator.generate()
        model = self.model.load_from_checkpoint(self.path)
        pred = model.predict(input)

        # if have multiple dim, only return first dim
        return output[:,:,0], pred[:,:,0]

class ShiftEvaluation(ModelEvaluation):

    def __init__(self, model, path, title='Shift Sequence') -> None:
        self.model = model
        self.path = path
        shift, length= map(lambda x:int(x), Path(self.path).stem.split('_')[1:])
        descrip = f'Shift = {shift}, Length = {length}.'
        self.generator = ShiftGenerator(size=20,shift=shift,length=length)

        super().__init__(model, path, title=title, descrip=descrip)

        self.trace1_name = 'Output'
        self.trace2_name = 'Prediction'

    def _generate_data(self):
            
        input, output = self.generator.generate()
        model = self.model.load_from_checkpoint(self.path)
        pred = model.predict(input)

        # if have multiple dim, only return first dim
        return output[:,:,0], pred[:,:,0]