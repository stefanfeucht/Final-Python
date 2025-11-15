import os
import pathlib
import pandas as pd
import dash
import plotly.express as px
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# ===========================
#   Inicializar App
# ===========================
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # necesario para Render
app.title = "Dashboard Financiero"

# ===========================
#   Cargar Dataset (Render-safe)
# ===========================
DATA_PATH = pathlib.Path(__file__).parent.joinpath("empresas.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ ERROR: No se encontró el archivo {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

sales_list = [
    "Total Revenues", "Cost of Revenues", "Gross Profit", "Total Operating Expenses",
    "Operating Income", "Net Income", "Shares Outstanding", "Close Stock Price",
    "Market Cap", "Multiple of Revenue"
]

# ===========================
#         LAYOUT
# ===========================
app.layout = html.Div([

    html.H2("Dashboard Financiero", style={"textAlign": "center"}),

    html.Div([
        html.Div(
            dcc.Dropdown(
                id="empresa-dropdown",
                value=["Apple", "Tesla", "Microsoft", "Google"],
                options=[{"label": x, "value": x} for x in sorted(df["Company"].unique())],
                multi=True,
                clearable=False
            ),
            style={"width": "50%"}
        ),

        html.Div(
            dcc.Dropdown(
                id="numericdropdown",
                value="Total Revenues",
                clearable=False,
                options=[{"label": x, "value": x} for x in sales_list]
            ),
            style={"width": "50%"}
        )
    ], style={"display": "flex", "gap": "10px"}),

    html.Br(),

    dcc.Graph(id="bar"),
    dcc.Graph(id="boxplot"),

    html.Div(id="table-container_1")
])

# ===========================
#        CALLBACKS
# ===========================
@app.callback(
    [
        Output("bar", "figure"),
        Output("boxplot", "figure"),
        Output("table-container_1", "children")
    ],
    [
        Input("empresa-dropdown", "value"),
        Input("numericdropdown", "value")
    ]
)
def display_value(selected_stock, selected_numeric):

    if not selected_stock or len(selected_stock) == 0:
        selected_stock = ["Amazon", "Tesla", "Microsoft", "Apple", "Google"]

    dfv_fltrd = df[df["Company"].isin(selected_stock)]

    fig = px.line(
        dfv_fltrd,
        color="Company",
        x="Quarter",
        y=selected_numeric,
        markers=True
    )

    fig.update_layout(
        title=f"{selected_numeric} de {', '.join(selected_stock)}",
        xaxis_title="Quarter"
    )

    fig2 = px.box(
        dfv_fltrd,
        color="Company",
        x="Company",
        y=selected_numeric
    )

    fig2.update_layout(title=f"{selected_numeric} - Distribución por Empresa")

    # Tabla
    df_reshaped = dfv_fltrd.pivot(index="Company", columns="Quarter", values=selected_numeric)
    df_reshaped2 = df_reshaped.reset_index()

    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df_reshaped2.columns],
        data=df_reshaped2.to_dict("records"),
        export_format="csv",
        style_cell={"fontSize": "12px"},
        style_header={"backgroundColor": "blue", "color": "white", "fontWeight": "bold"}
    )

    return fig, fig2, table

# ===========================
#       RUN SERVER
# ===========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
