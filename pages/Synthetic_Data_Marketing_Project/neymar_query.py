import os
import re
from dash import Dash, html, dcc, Input, Output
import dash_table
from google.cloud import bigquery

# -----------------------------------------------------------------------------
# CONFIG — update as needed
# -----------------------------------------------------------------------------
# If running locally, uncomment and set your service account:
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/creds.json"

PROJECT = "mindful-vial-460001-h6"
DATASET = "euphoria"

# -----------------------------------------------------------------------------
# CLIENT SETUP
# -----------------------------------------------------------------------------
client = bigquery.Client(project=PROJECT)

# -----------------------------------------------------------------------------
# DASH APP
# -----------------------------------------------------------------------------
app = Dash(__name__)
app.layout = html.Div([
    html.H2("BigQuery SQL Runner"),
    dcc.Textarea(
        id='sql-input',
        value=(
            f"SELECT *\n"
            f"FROM `{PROJECT}.{DATASET}.streams_topic`\n"
            "LIMIT 5;"
        ),
        style={'width': '100%', 'height': '120px'}
    ),
    html.Button("Run Query", id='run-btn', n_clicks=0, style={'marginTop': '10px'}),
    html.Div(id='table-container', style={'marginTop': '20px'})
])

@app.callback(
    Output('table-container', 'children'),
    Input('run-btn', 'n_clicks'),
    Input('sql-input', 'value')
)
def render_table(n_clicks, query):
    # Don't run until button clicked
    if not n_clicks:
        return html.Div("Enter a SQL query above and click ▶️ Run Query")

    # Replace placeholders if used
    q = re.sub(r"\bPROJECT\b", PROJECT, query)
    q = re.sub(r"\bDATASET\b", DATASET, q)

    # Execute and fetch DataFrame
    df = client.query(q).to_dataframe(create_bqstorage_client=False)

    # Render Dash DataTable
    return dash_table.DataTable(
        columns=[{'name': c, 'id': c} for c in df.columns],
        data=df.to_dict('records'),
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_header={'fontWeight': 'bold'},
        style_cell={'textAlign': 'left'}
    )

if __name__ == "__main__":
    app.run(debug=True)
