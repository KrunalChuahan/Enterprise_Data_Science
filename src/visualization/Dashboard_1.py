# required packages
import pandas as pd
import numpy as np
import dash
print('Your current dash board version is:' + dash.__version__)
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import plotly.graph_objects as go
import os

# import local CSV file
df_input_large=pd.read_csv('data/processed/COVID_final_set.csv',sep=';')

# for plotting
fig = go.Figure()
# for dashboard development
app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    # TU_KL_Data Science_COVID-19_Task_1

    * Aim: To learn data science by applying a cross-industry standard process.

    * X-axis:- Timeline in Days,
      Y-axis:- confirmed infected cases in the log-scale format

    ### The first dropdown menu enables selection of one or multiple  countries for visualization.
    ### The seconds dropdown menu contains four option:
        1. The ‘Timeline Confirmed’ represents confirmed infected cases along the timeline.
        2. The ‘Timeline Confirmed Filtered’ represents filtered (after applying sav-gol filter)confirmed infected cases along the timeline.
        3. The ‘Timeline Doubling Rate’ represents the doubling rate calculation on the infected cases along the timeline from the first option.
        4. The ‘Timeline Doubling Rate Filtered’ represents the doubling rate calculation on the infected cases along the timeline from the second option.

    '''),

    dcc.Markdown('''
    ## Select the Country for visualization
    '''),

    dcc.Dropdown(
        id='country_drop_down',
        options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
        value=['Germany','US','Brazil'], # which are pre-selected in default layout
        multi=True
    ),

    dcc.Markdown('''
        ## Select Timeline of confirmed COVID-19 cases or the approximated doubling time
        '''),


    dcc.Dropdown(
    id='doubling_time',
    options=[
        {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
        {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
        {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
        {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
    ],
    value='confirmed',multi=False),dcc.Graph(figure=fig, id='main_window_slope')])

@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
    Input('doubling_time', 'value')])
def update_figure_layout(country_list,show_doubling):
    if 'doubling_rate' in show_doubling:
        my_yaxis={'type':"log",
               'title':'Approximated doubling rate over 3 days (larger numbers are better #stayathome)'
          }
    else:
        my_yaxis={'type':"log",
                  'title':'Confirmed infected people (From johns hopkins csse, log-scale)'}
    traces = []
    for each in country_list:
        df_plot=df_input_large[df_input_large['country']==each]

        if show_doubling=='doubling_rate_filtered':
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.mean).reset_index()
        else:
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.sum).reset_index()

        traces.append(dict(x=df_plot.date,
                                y=df_plot[show_doubling],
                                mode='markers+lines',opacity=1.0,name=each))

    return {
            'data': traces,
            'layout': dict (
                width=1000,height=720,
                xaxis={'title':'Timeline in the days','tickangle':-45,'nticks':20,
                        'tickfont':dict(size=14,color="#0c6887"),},yaxis=my_yaxis)}

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
