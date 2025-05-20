import os
from datetime import datetime
from google.cloud import bigquery

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# ── 1) GCP / BigQuery setup ─────────────────────────────────────────────
PROJECT = "mindful-vial-460001-h6"
DATASET = "euphoria"
bq = bigquery.Client(project=PROJECT)

# ── 2) Build list of the last 10 years for the dropdown ─────────────────
current_year = datetime.utcnow().year
YEARS = list(range(current_year - 9, current_year + 1))[::-1]  # e.g. [2025,2024,…,2016]

# ── 3) Dash app ────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Yearly Relative Watch Hours"

app.layout = html.Div([
    html.H1("Watch Hours by Country (Relative Rank)"),
    html.Div([
      html.Label("Select Year:"),
      dcc.Dropdown(
          id='year-dropdown',
          options=[{'label': str(y), 'value': y} for y in YEARS],
          value=current_year,
          clearable=False,
          style={'width':'120px'}
      )
    ], style={'marginBottom':'1em'}),
    dcc.Graph(id='choropleth'),
])

# ── 4) Callback: fetch & plot the relative percent‐rank choropleth ────────
@app.callback(
    Output('choropleth', 'figure'),
    Input('year-dropdown', 'value'),
)
def update_choropleth(selected_year):
    # 1) Compute each country's watch_hours and its percentile rank within that year
    sql = f"""
    WITH country_totals AS (
      SELECT
        country,
        SUM(length)/3600.0 AS watch_hours
      FROM `{PROJECT}.{DATASET}.watch_topic`
      WHERE EXTRACT(YEAR FROM DATE(date)) = {selected_year}
      GROUP BY country
    )
    SELECT
      country,
      watch_hours,
      PERCENT_RANK() OVER (ORDER BY watch_hours) AS pct_rank
    FROM country_totals
    """
    df = bq.query(sql).to_dataframe()

    # 2) If no data, show empty
    if df.empty:
        return px.choropleth(
            locations=[],
            color=[]
        )

    # 3) Plot using pct_rank as the color
    fig = px.choropleth(
        df,
        locations="country",
        locationmode="country names",
        color="pct_rank",
        hover_name="country",
        hover_data={"watch_hours":":.1f", "pct_rank":":.2f"},
        color_continuous_scale="Viridis",
        range_color=(0,1),
        labels={"pct_rank":"Relative rank","watch_hours":"Hours"},
        title=f"Relative Watch Hours in {selected_year}"
    )
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        coloraxis_colorbar={"title":"Percentile"}
    )
    return fig

# ── 5) Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
