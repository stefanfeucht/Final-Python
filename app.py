# ======================================================
#                FINAL PYTHON — DASH APP
# ======================================================

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, norm

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# ------------------------------------------------------
# INICIALIZAR APP
# ------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server     # <- Para Render

# ------------------------------------------------------
# CARGAR ARCHIVOS DESDE EL REPO
# ------------------------------------------------------
EXCEL_PATH = "Final.xlsx"
CRYPTO_PATH = "Crypto_historical_data.csv.gz"

# Leer hojas del Excel
P_weeklyS = pd.read_excel(EXCEL_PATH, sheet_name="P WeeklyS")
R_weeklyS = pd.read_excel(EXCEL_PATH, sheet_name="R WeeklyS")

P_weeklyC = pd.read_excel(EXCEL_PATH, sheet_name="P WeeklyC")
R_weeklyC = pd.read_excel(EXCEL_PATH, sheet_name="R WeeklyC")

# ------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------
def fix_date(df):
    col = [c for c in df.columns if c.lower() in ("date", "fecha")][0]
    df.rename(columns={col: "Date"}, inplace=True)

    if np.issubdtype(df["Date"].dtype, np.number):
        df["Date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df["Date"], "D")
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df.dropna(subset=["Date"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

P_weeklyS = fix_date(P_weeklyS)
R_weeklyS = fix_date(R_weeklyS)
P_weeklyC = fix_date(P_weeklyC)
R_weeklyC = fix_date(R_weeklyC)

# ------------------------------------------------------
# TICKERS ACCIONES & CRIPTOS
# ------------------------------------------------------
stocks = [c for c in P_weeklyS.columns if c != "Date"]
cryptos = [c for c in P_weeklyC.columns if c != "Date"]

# ------------------------------------------------------
# LAYOUT — 3 TABS
# ------------------------------------------------------
app.layout = dbc.Container([

    html.H1("Dashboard Financiero — Final Python", className="text-center mt-4 mb-4"),

    dcc.Tabs(id="tabs", value="tab-1", children=[
        dcc.Tab(label="📈 Acciones (Precios y Retornos)", value="tab-1"),
        dcc.Tab(label="📊 Distribuciones y Métricas", value="tab-2"),
        dcc.Tab(label="💰 Criptomonedas", value="tab-3"),
    ]),

    html.Div(id="tabs-content", className="mt-4")

], fluid=True)


# ------------------------------------------------------
# CALLBACK PRINCIPAL DE TABS
# ------------------------------------------------------
@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value")
)
def render_tabs(tab):
    if tab == "tab-1":
        return layout_tab1()
    elif tab == "tab-2":
        return layout_tab2()
    elif tab == "tab-3":
        return layout_tab3()


# ------------------------------------------------------
# TAB 1 — ACCIONES
# ------------------------------------------------------
def layout_tab1():
    return html.Div([

        html.Label("Elige acción:"),
        dcc.Dropdown(
            id="stock-dropdown",
            options=[{"label": s, "value": s} for s in stocks],
            value=stocks[0],
            clearable=False
        ),

        html.Label("Tipo de gráfica:", className="mt-3"),
        dcc.Dropdown(
            id="tipo-dropdown",
            options=[
                {"label": "Precio", "value": "P"},
                {"label": "Retorno", "value": "R"}
            ],
            value="P",
            clearable=False
        ),

        dcc.Graph(id="graph-stock", className="mt-4")
    ])


@app.callback(
    Output("graph-stock", "figure"),
    [Input("stock-dropdown", "value"),
     Input("tipo-dropdown", "value")]
)
def update_stock_graph(stock, tipo):
    df = P_weeklyS if tipo == "P" else R_weeklyS
    fig = px.line(df, x="Date", y=stock, title=f"{stock} — {'Precios' if tipo=='P' else 'Retornos'}")
    fig.update_layout(template="plotly_white")
    return fig


# ------------------------------------------------------
# TAB 2 — DISTRIBUCIONES
# ------------------------------------------------------
def layout_tab2():
    return html.Div([

        html.Label("Elige acción:"),
        dcc.Dropdown(
            id="dist-stock",
            options=[{"label": s, "value": s} for s in stocks],
            value=stocks[0],
            clearable=False
        ),

        dcc.Graph(id="hist-dist", className="mt-4"),

        html.H3("📉 Métricas de Riesgo", className="mt-4"),
        dash_table.DataTable(id="tabla-metricas")
    ])


def calc_metrics(series):
    return {
        "Media": round(series.mean(), 5),
        "Desv.Std": round(series.std(), 5),
        "Curtosis": round(kurtosis(series, fisher=False), 5),
        "Skewness": round(skew(series), 5),
        "VaR 95%": round(np.percentile(series, 5), 5),
        "CVaR 95%": round(series[series <= np.percentile(series, 5)].mean(), 5),
        "Sharpe": round(series.mean() * 52 / (series.std() * np.sqrt(52)), 5)
    }


@app.callback(
    [Output("hist-dist", "figure"),
     Output("tabla-metricas", "data"),
     Output("tabla-metricas", "columns")],
    Input("dist-stock", "value")
)
def update_distribution(stock):

    series = R_weeklyS[stock].dropna()

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=series, nbinsx=50, name="Histograma"))
    x_vals = np.linspace(series.min(), series.max(), 200)
    fig.add_trace(go.Scatter(x=x_vals,
                             y=norm.pdf(x_vals, series.mean(), series.std()),
                             mode="lines", name="Densidad"))
    fig.update_layout(template="plotly_white",
                      title=f"Distribución — {stock}")

    m = calc_metrics(series)
    columns = [{"name": k, "id": k} for k in m.keys()]
    data = [m]

    return fig, data, columns


# ------------------------------------------------------
# TAB 3 — CRIPTOS
# ------------------------------------------------------
def layout_tab3():

    example_crypto = cryptos[0]

    fig = px.line(P_weeklyC, x="Date", y=example_crypto,
                  title=f"Precio histórico — {example_crypto}")

    return html.Div([
        dcc.Graph(figure=fig)
    ])


# ------------------------------------------------------
# RUN
# ------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
