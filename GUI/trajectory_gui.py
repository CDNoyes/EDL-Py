""" Kivy based GUI development.
    Installation required some additional upgrades/installations 
    Additionally, garden components have to be installed separately like:
    garden install graph
    
    Simulation.GUI should save its data somewhere (possibly temporary) and call this GUI 
"""
# Basic kivy imports 
from kivy.app import App 

from kivy.uix.button import Button
from kivy.uix.widget import Widget 
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.spinner import Spinner
from kivy.uix.checkbox import CheckBox

# from kivy.graphics import Color, Ellipse, Line 

# from kivy.vector import Vector
# from kivy.properties import NumericProperty, ObjectProperty, ReferenceListProperty
# from kivy.clock import Clock 

# garden extension imports 
# from kivy.garden.graph import Graph, LinePlot, SmoothLinePlot

import matplotlib
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
from kivy.garden.matplotlib import FigureCanvasKivyAgg, NavigationToolbar2Kivy

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import os 

if False:
    plt.style.use('dark_background')   
    main_plot_color = 'y'
else:
    main_plot_color = 'k'
    
# plt.style.use('ggplot')   
   
class TrajectoryGUIApp(App):

    km2m = 1000
    m2km = 1e-3 
    rad2deg = 180.0/np.pi

    def build(self):
    
        layout = FloatLayout(source='./space.jpg')
        # self.data = generate_dataset()
        self.data = pd.read_pickle('./results/entry_test_data.pkl')
        file_list = self.data.columns.tolist()
        
        file_dropdown_x = Spinner(text=file_list[0], values=file_list, size_hint=(0.15,0.06),  pos_hint={'right':0.85,'top':.92}, sync_height=True)
        file_dropdown_y = Spinner(text=file_list[1], values=file_list, size_hint=(0.15,0.06),  pos_hint={'right':1,'top':0.92}, sync_height=True)

        file_dropdown_x.bind(text=self.new_plot)
        file_dropdown_y.bind(text=self.new_plot)
        
        self.x = file_dropdown_x
        self.y = file_dropdown_y
        
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)
        self.ax = ax 
        ax.grid()
        
        canvas = FigureCanvasKivyAgg(figure=fig,size_hint=(0.7,0.92),pos_hint={'right':0.7})
        self.canvas = canvas
        
        nav = NavigationToolbar2Kivy(canvas)
        
        
        # Additional functionality 
        reset = Button(text='Clear Figure',size_hint=(0.15,0.06),pos_hint={'right':0.5,'top':0.99},background_color=[1,0,0,1])
        reset.bind(on_press=self.reset_plot)
        
        hold = Button(text='Hold',size_hint=(0.15,0.06),pos_hint={'right':0.6,'top':1},background_color=[1,0,1,1])
        # hold.bind(on_press=rebind) 
        
        # Add everything to the layout 
        layout.add_widget(file_dropdown_x)
        layout.add_widget(file_dropdown_y)
        layout.add_widget(nav.actionbar)
        layout.add_widget(canvas)
        layout.add_widget(reset)
        # layout.add_widget(hold)
        
        self.update_plot(None,None)

        return layout


    def reset_plot(self,button):
        self.ax.lines=[]
        plt.xlabel('')
        plt.ylabel('')
        self.ax.set_ylim(0, 1)
        self.canvas.draw()
        return 
        

    def new_plot(self, *args, **kwargs):
        # Clear the figure and draw the new line 
        self.ax.lines=[] 
        self.update_plot()
        
    def update_plot(self, *args,**kwargs):
        
        self.ax.plot(self.data[self.x.text],self.data[self.y.text],main_plot_color,lineWidth=3)    
        
        plt.xlabel(self.x.text)
        plt.ylabel(self.y.text)
        
        # Set sane limits 
        sxmin = np.sign(self.data[self.x.text].min())
        sxmax = np.sign(self.data[self.x.text].max())
        symin = np.sign(self.data[self.y.text].min())
        symax = np.sign(self.data[self.y.text].max())
        d = 0.1 # fraction of difference 
        self.ax.set_xlim(self.data[self.x.text].min()*(1-d*sxmin), self.data[self.x.text].max()*(1+d*sxmax))
        self.ax.set_ylim(self.data[self.y.text].min()*(1-d*symin), self.data[self.y.text].max()*(1+d*symax))
        self.canvas.draw()
        return


def generate_dataset():
    import sys
    from os import path 
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
    
    from EntryGuidance.Simulation import Simulation, EntrySim
    from EntryGuidance.InitialState import InitialState
    from EntryGuidance.HPC import profile 
    
    reference_sim = Simulation(output=False,**EntrySim())
    banks = [-np.pi/2, np.pi/2,-np.pi/9]
    bankProfile = lambda **d: profile(d['time'],[89.3607, 136.276], banks)                                
    x0 = InitialState()
    output_ref = reference_sim.run(x0,[bankProfile],StepsPerCycle=10)
    reference_sim.df.to_pickle('./results/entry_test_data.pkl')
    return reference_sim.df
    
    
if __name__ == "__main__":      
    # generate_dataset()
    TrajectoryGUIApp().run()
    