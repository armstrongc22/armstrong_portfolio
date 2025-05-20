import os
from google.cloud import bigquery
import pandas as pd

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# ── 1) Set up your GCP credentials & BQ client ────────────────────────────────
# either set GOOGLE_APPLICATION_CREDENTIALS in your env, or do it in code:
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your-key.json"
PROJECT = "mindful-vial-460001-h6"
DATASET = "euphoria"

bq = bigquery.Client(project=PROJECT)

# ── 2) Dash app boilerplate ────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Live Watch Hours by Country"

app.layout = html.Div([
    html.H1("Live Watch Hours by Country (last 5 minutes)", style={"textAlign":"center"}),
    dcc.Graph(id="live-choropleth"),
    dcc.Interval(id="interval", interval=10*1000, n_intervals=0)  # every 10 seconds
])

# ── 3) Callback to refresh the map ─────────────────────────────────────────────
@app.callback(
    Output("live-choropleth", "figure"),
    Input("interval", "n_intervals")
)
def update_map(n):
    sql = f"""
      SELECT
        country,
        SUM(length) / 3600.0 AS watch_hours
      FROM `{PROJECT}.{DATASET}.watch_topic`
      WHERE date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 MINUTE)
      GROUP BY country
    """
    df = bq.query(sql).to_dataframe()
    if df.empty:
        # show an empty frame
        return px.choropleth(
            locations=[],
            locationmode="country names",
            color=[],
            title="No data yet"
        )
    fig = px.choropleth(
        df,
        locations="country",
        locationmode="country names",
        color="watch_hours",
        hover_name="country",
        color_continuous_scale="Viridis",
        range_color=(0, df.watch_hours.max()),
        labels={"watch_hours":"Hours watched"}
    )
    fig.update_layout(
        margin={"r":0,"t":30,"l":0,"b":0},
        coloraxis_colorbar={"title":"Hours"}
    ) 
    return fig

# ── 4) Run it ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
