


import os

from google.cloud import bigquery
import pandas as pd

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

# ── 1) GCP / BigQuery setup ─────────────────────────────────────────────
# adjust this to wherever your JSON key lives
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mindful-vial-460001-h6-4d83b36dd3e9.json"

PROJECT = "mindful-vial-460001-h6"
DATASET = "euphoria"
client = bigquery.Client(project=PROJECT)

# ── 2) One-shot “all KPIs” union query ────────────────────────────────────
SQL = f"""
WITH
  viewed AS (
    SELECT country, SUM(length) AS total_watch_seconds
    FROM `{PROJECT}.{DATASET}.watch_topic`
    GROUP BY country
    ORDER BY total_watch_seconds DESC
    LIMIT 10
  ),
  purchased AS (
    SELECT product_name, COUNT(*) AS purchase_count
    FROM `{PROJECT}.{DATASET}.purchase_events_topic`
    GROUP BY product_name
    ORDER BY purchase_count DESC
    LIMIT 8
  ),
  streamer_perf AS (
    SELECT
      p.screen_name,
      SUM((s.viewers_total/NULLIF(s.length,0)) * s.comments_total) AS performance_score
    FROM `{PROJECT}.{DATASET}.streams_topic`   AS s
    JOIN `{PROJECT}.{DATASET}.partners_topic`  AS p
      ON s.partner_id = p.partner_id
    GROUP BY p.screen_name
    ORDER BY performance_score DESC
    LIMIT 10
  ),
  best_games AS (
    SELECT product_name, COUNT(*) AS purchase_count
    FROM `{PROJECT}.{DATASET}.purchase_events_topic`
    WHERE category = 'game'
    GROUP BY product_name
    ORDER BY purchase_count DESC
    LIMIT 2
  ),
  streamed_games AS (
    SELECT
      g.title      AS game_title,
      COUNT(*)     AS stream_count
    FROM `{PROJECT}.{DATASET}.streams_topic` AS s
    JOIN `{PROJECT}.{DATASET}.games_topic`   AS g
      ON s.game_id = g.game_id
    GROUP BY g.title
    ORDER BY stream_count DESC
    LIMIT 2
  ),
  top_cust_purch AS (
    SELECT customer_id, COUNT(*) AS purchase_count
    FROM `{PROJECT}.{DATASET}.purchase_events_topic`
    GROUP BY customer_id
    ORDER BY purchase_count DESC
    LIMIT 1000
  ),
  top_cust_watch AS (
    SELECT customer_id, SUM(length) AS total_watch_seconds
    FROM `{PROJECT}.{DATASET}.watch_topic`
    GROUP BY customer_id
    ORDER BY total_watch_seconds DESC
    LIMIT 1000
  ),
  monthly_watch AS (
    SELECT
      FORMAT('%04d-%02d',
        EXTRACT(YEAR  FROM DATE(date)),
        EXTRACT(MONTH FROM DATE(date))
      ) AS period,
      SUM(length) AS total_watch_seconds
    FROM `{PROJECT}.{DATASET}.watch_topic`
    GROUP BY period
    ORDER BY period
  ),
  yearly_watch AS (
    SELECT
      CAST(EXTRACT(YEAR FROM DATE(date)) AS STRING) AS period,
      SUM(length) AS total_watch_seconds
    FROM `{PROJECT}.{DATASET}.watch_topic`
    GROUP BY period
    ORDER BY period
  ),
  monthly_merch AS (
    SELECT
      FORMAT('%04d-%02d',
        EXTRACT(YEAR  FROM DATE(timestamp)),
        EXTRACT(MONTH FROM DATE(timestamp))
      ) AS period,
      SUM(price) AS total_merch_sales
    FROM `{PROJECT}.{DATASET}.purchase_events_topic`
    WHERE category = 'merch'
    GROUP BY period
    ORDER BY period
  ),
  yearly_merch AS (
    SELECT
      CAST(EXTRACT(YEAR FROM DATE(timestamp)) AS STRING) AS period,
      SUM(price) AS total_merch_sales
    FROM `{PROJECT}.{DATASET}.purchase_events_topic`
    WHERE category = 'merch'
    GROUP BY period
    ORDER BY period
  )

SELECT 'Top 10 Viewed Countries'       AS kpi, country        AS label, CAST(total_watch_seconds AS STRING)     AS value FROM viewed
UNION ALL
SELECT 'Top 8 Purchased Products'      AS kpi, product_name   AS label, CAST(purchase_count            AS STRING) AS value FROM purchased
UNION ALL
SELECT 'Top 10 Streamer Performance'   AS kpi, screen_name    AS label, CAST(ROUND(performance_score,2) AS STRING) AS value FROM streamer_perf
UNION ALL
SELECT 'Top 2 Best-Selling Games'      AS kpi, product_name   AS label, CAST(purchase_count            AS STRING) AS value FROM best_games
UNION ALL
SELECT 'Top 2 Most-Streamed Games'     AS kpi, game_title     AS label, CAST(stream_count              AS STRING) AS value FROM streamed_games
UNION ALL
SELECT 'Top 1000 Customers by Purchases'      AS kpi, customer_id AS label, CAST(purchase_count            AS STRING) AS value FROM top_cust_purch
UNION ALL
SELECT 'Top 1000 Customers by Watch Seconds'  AS kpi, customer_id AS label, CAST(total_watch_seconds       AS STRING) AS value FROM top_cust_watch
UNION ALL
SELECT 'Monthly Watch (sec)'           AS kpi, period         AS label, CAST(total_watch_seconds       AS STRING) AS value FROM monthly_watch
UNION ALL
SELECT 'Yearly Watch (sec)'            AS kpi, period         AS label, CAST(total_watch_seconds       AS STRING) AS value FROM yearly_watch
UNION ALL
SELECT 'Monthly Merch Sales'           AS kpi, period         AS label, FORMAT('$%.2f', total_merch_sales)        AS value FROM monthly_merch
UNION ALL
SELECT 'Yearly Merch Sales'            AS kpi, period         AS label, FORMAT('$%.2f', total_merch_sales)        AS value FROM yearly_merch
;
"""

# load it once
df = client.query(SQL).to_dataframe()

# ── 3) Dash app ─────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Euphoria Dynamic KPI Dashboard"

app.layout = html.Div([
    html.H1("Euphoria KPI Dashboard"),
    dcc.Dropdown(
        id="kpi-dropdown",
        options=[{"label": k, "value": k} for k in df["kpi"].unique()],
        value=df["kpi"].unique()[0],
        clearable=False,
        style={"width":"60%", "marginBottom":"1em"}
    ),
    dash_table.DataTable(
        id="kpi-table",
        columns=[{"name": c, "id": c} for c in df.columns],
        data=[],
        page_size=20,
        style_table={"overflowX":"auto"},
        style_header={"fontWeight":"bold"},
        style_cell={"textAlign":"left", "padding":"5px"}
    )
], style={"padding":"20px"})


@app.callback(
    Output("kpi-table", "data"),
    Input("kpi-dropdown", "value")
)
def filter_table(selected_kpi):
    dff = df[df["kpi"] == selected_kpi]
    return dff.to_dict("records")


if __name__ == "__main__":
    app.run(debug=True)






