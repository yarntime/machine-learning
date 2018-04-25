import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
import plotly.plotly as py
from plotly.graph_objs import *
from scipy.stats import rayleigh
from flask import Flask
import numpy as np
import pandas as pd
import os
import random

app = dash.Dash('Abnormal Detection')

df = pd.read_csv("300_data.csv")

app.layout = html.Div([
    html.Div([
        html.H2("Abnormal Detection"),
        html.Img(src="./images/ai.jpg"),
    ], className='banner'),
    html.Div([
        html.Div([
            html.H3("KPI")
        ], className='Title'),
        html.Div([
            dcc.Graph(id='kpi-compare'),
        ], className='two columns kip'),
        dcc.Interval(id='kpi-update', interval=1000, n_intervals=0),
    ], className='row wind-speed-row')
], style={'padding': '0px 10px 15px 10px',
           'marginLeft': 'auto', 'marginRight': 'auto', "width": "900px",
           'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)'})

@app.callback(Output('kpi-compare', 'figure'), [Input('kpi-update', 'n_intervals')])
def gen_wind_speed(interval):
    global df
    df.head()

    rows = random.sample(list(df.index), 200)
    df_shows = df.ix[rows]

    trace = Scatter(
        y=df_shows['y'],
        line=Line(
            color='#42C4F7'
        ),
        hoverinfo='skip',
        error_y=ErrorY(
            type='data',
            array=df_shows['y'],
            thickness=1.5,
            width=2,
            color='#B4E8FC'
        ),
        mode='lines'
    )

    layout = Layout(
        height=450,
        xaxis=dict(
            range=[0, 200],
            showgrid=False,
            showline=False,
            zeroline=False,
            fixedrange=True,
            tickvals=[0, 50, 100, 150, 200],
            ticktext=['200', '150', '100', '50', '0'],
            title='Date'
        ),
        yaxis=dict(
            range=[0, 5000],
            showline=False,
            fixedrange=True,
            zeroline=False,
            tickvals=[0, 1000, 2000, 3000, 4000, 5000],
            ticktext=['200', '150', '100', '50', '0'],
            nticks=max(4500, round(df['y'].iloc[-1]/10))
        ),
        margin=Margin(
            t=45,
            l=50,
            r=50
        )
    )

    return Figure(data=[trace], layout=layout)

if __name__ == '__main__':
    app.run_server(port=8888)