import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import skew, kurtosis, norm
import plotly.express as px
import dash
from dash import Dash
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
#Exploración de datos de acciones

# Paso 1: Inicializar la app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#Github:agregar linea de server que usa github
server=app.server

app.title = "Final Pithon"
app.title = "Acciones"

path = "Final.xlsx"
P_weeklyS = pd.read_excel(path, sheet_name="P WeeklyS")
R_weeklyS = pd.read_excel(path, sheet_name="R WeeklyS")

def fix_date_column(df):
    # detectar la columna que parece fecha
    for col in df.columns:
        if col.lower() in ("date", "fecha"):
            df = df.rename(columns={col: "Date"})
            break
    if "Date" not in df.columns:
        raise ValueError("No se encontró columna llamada 'Date' o 'fecha'.")

    # Si las fechas son números tipo 45257 (formato serial de Excel)
    if pd.api.types.is_numeric_dtype(df["Date"]):
        # Excel usa el día 1899-12-30 como base
        df["Date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df["Date"], unit="D")
    else:
        # Si es texto, intentar parsearlo
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)

    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df

P_weeklyS = fix_date_column(P_weeklyS)
R_weeklyS = fix_date_column(R_weeklyS)

# Confirmar rango de fechas
min_date = min(P_weeklyS["Date"].min(), R_weeklyS["Date"].min())
max_date = max(P_weeklyS["Date"].max(), R_weeklyS["Date"].max())
print("✅ Rango de fechas corregido:", min_date.date(), "->", max_date.date())

# === TICKERS ===
stocks = [c for c in P_weeklyS.columns if c != "Date"]

# === CREAR GRÁFICO INTERACTIVO ===
fig = go.Figure()

# Precios
for stock in stocks:
    fig.add_trace(go.Scatter(
        x=P_weeklyS["Date"], y=P_weeklyS[stock],
        mode='lines', name=f"{stock} (Precio)", visible=True
    ))

# Retornos
for stock in stocks:
    fig.add_trace(go.Scatter(
        x=R_weeklyS["Date"], y=R_weeklyS[stock],
        mode='lines', name=f"{stock} (Retorno)", visible=False
    ))

n = len(stocks)

# Dropdowns
buttons_tipo = [
    dict(label="Precios", method="update",
         args=[{"visible": [True]*n + [False]*n},
               {"title": "Precios Semanales"}]),
    dict(label="Retornos", method="update",
         args=[{"visible": [False]*n + [True]*n},
               {"title": "Retornos Semanales"}])
]

buttons_acciones = [dict(label="Todas", method="update",
                         args=[{"visible": [True]*n + [False]*n},
                               {"title": "Todas las acciones"}])]

for i, stock in enumerate(stocks):
    vis = [False]*(2*n)
    vis[i] = True
    buttons_acciones.append(dict(label=stock, method="update",
                                 args=[{"visible": vis},
                                       {"title": f"{stock}"}]))

fig.update_layout(
    updatemenus=[
        dict(active=0, buttons=buttons_tipo, x=0, y=1.15),
        dict(active=0, buttons=buttons_acciones, x=0.25, y=1.15)
    ],
    xaxis=dict(title="Fecha", type="date", rangeslider=dict(visible=True)),
    yaxis_title="Valor",
    title="Precios y Retornos de Acciones",
    template="plotly_white"
)

fig.update_xaxes(range=[min_date, max_date], tickformat="%Y-%m-%d")

fig.show()




#Distribuciones y métricas

# === FUNCIONES AUXILIARES ===
def fix_date_column(df):
    for col in df.columns:
        if col.lower() in ("date", "fecha"):
            df = df.rename(columns={col: "Date"})
            break
    if "Date" not in df.columns:
        raise ValueError("No se encontró columna llamada 'Date' o 'fecha'.")

    if pd.api.types.is_numeric_dtype(df["Date"]):
        df["Date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df["Date"], unit="D")
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

def historical_var(series, level=0.95):
    return np.percentile(series.dropna(), 100 * (1 - level))

def cvar_historical(series, level=0.95):
    s = series.dropna()
    var = historical_var(s, level)
    tail = s[s <= var]
    return tail.mean() if len(tail) > 0 else var

def sharpe_ratio(series, rf=0.0, periods_per_year=52):
    s = series.dropna()
    if s.std(ddof=1) == 0 or len(s) == 0:
        return np.nan
    ann_ret = s.mean() * periods_per_year
    ann_vol = s.std(ddof=1) * np.sqrt(periods_per_year)
    return (ann_ret - rf) / ann_vol

# === CARGA DE DATOS ===
path = "Final.xlsx"
R_weeklyS = pd.read_excel(path, sheet_name="R WeeklyS")
R_weeklyS = fix_date_column(R_weeklyS)

# === RANGO DE 3 AÑOS ===
end_date = R_weeklyS["Date"].max()
start_date = end_date - pd.DateOffset(years=3)
R3 = R_weeklyS[(R_weeklyS["Date"] >= start_date) & (R_weeklyS["Date"] <= end_date)].copy()

stocks = [c for c in R3.columns if c != "Date"]

# === CALCULAR MÉTRICAS DE RIESGO ===
metrics = []
for s in stocks:
    series = R3[s].dropna()
    metrics.append({
        "Acción": s,
        "Media": series.mean(),
        "Desv.Std": series.std(),
        "Curtosis": kurtosis(series, fisher=False),
        "Skewness": skew(series),
        "VaR 95%": historical_var(series, 0.95),
        "VaR 90%": historical_var(series, 0.90),
        "CVaR 95%": cvar_historical(series, 0.95),
        "Sharpe (anual)": sharpe_ratio(series)
    })
df_metrics = pd.DataFrame(metrics)
print("📈 Métricas de riesgo (últimos 3 años):")
print(df_metrics.round(5))

# === GRÁFICO INTERACTIVO ===
fig = go.Figure()

for s in stocks:
    series = R3[s].dropna()
    fig.add_trace(go.Histogram(
        x=series, nbinsx=50, name=s,
        opacity=0.7, visible=(s == stocks[0])
    ))
    # Curva de densidad
    x_vals = np.linspace(series.min(), series.max(), 200)
    y_vals = norm.pdf(x_vals, series.mean(), series.std()) * len(series) * (series.max()-series.min())/50
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode='lines',
        name=f"{s} (densidad)", visible=(s == stocks[0])
    ))

# === BOTÓN DROPDOWN ===
buttons = []
for i, s in enumerate(stocks):
    vis = [False] * (2 * len(stocks))
    vis[i*2] = True
    vis[i*2 + 1] = True
    buttons.append(dict(label=s, method="update",
                        args=[{"visible": vis},
                              {"title": f"Distribución de retornos - {s}"}]))

fig.update_layout(
    updatemenus=[dict(active=0, buttons=buttons, x=0, y=1.15)],
    title=f"Distribución de retornos semanales (últimos 3 años) - {stocks[0]}",
    xaxis_title="Retorno semanal",
    yaxis_title="Frecuencia (histograma + densidad normal)",
    barmode="overlay",
    template="plotly_white"
)

fig.show()






# === 1️⃣ CARGA Y LIMPIEZA DE DATOS ===

# Cambia la ruta si es necesario
P_weeklyC = pd.read_excel("Final.xlsx", sheet_name="P WeeklyC")
R_weeklyC = pd.read_excel("Final.xlsx", sheet_name="R WeeklyC ")

for df in [P_weeklyC, R_weeklyC]:
    # Detectar y renombrar la columna de fecha
    date_col = [c for c in df.columns if c.lower() in ("date", "fecha")][0]
    df.rename(columns={date_col: "Date"}, inplace=True)

    # Convertir fechas (soporta números tipo Excel, strings, etc.)
    if np.issubdtype(df["Date"].dtype, np.number):
        df["Date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df["Date"], "D")
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df.dropna(subset=["Date"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

# Detectar rango de fechas real
min_date = min(P_weeklyC["Date"].min(), R_weeklyC["Date"].min())
max_date = max(P_weeklyC["Date"].max(), R_weeklyC["Date"].max())
print(f"✅ Rango de fechas detectado: {min_date.date()} → {max_date.date()}")

cryptos = [c for c in R_weeklyC.columns if c != "Date"]

# =============================================================================
# === 2️⃣ GRÁFICO DE BOLLINGER SOBRE RETORNOS ===
# =============================================================================

window = 20
fig = make_subplots(rows=len(cryptos), cols=1, shared_xaxes=True,
                    subplot_titles=[f"{c}" for c in cryptos])

for i, crypto in enumerate(cryptos, start=1):
    r = R_weeklyC.set_index("Date")[crypto].dropna()
    if r.empty:
        continue

    roll_mean = r.rolling(window=window).mean()
    roll_std = r.rolling(window=window).std()
    upper = roll_mean + 2 * roll_std
    lower = roll_mean - 2 * roll_std

    fig.add_trace(go.Scatter(x=r.index, y=r, mode="lines", name=f"{crypto} Retornos", line=dict(color="blue")), row=i, col=1)
    fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean, name=f"{crypto} Media móvil (20)", line=dict(color="orange")), row=i, col=1)
    fig.add_trace(go.Scatter(x=upper.index, y=upper, name=f"{crypto} +2σ", line=dict(color="green", dash="dot")), row=i, col=1)
    fig.add_trace(go.Scatter(x=lower.index, y=lower, name=f"{crypto} -2σ", line=dict(color="red", dash="dot")), row=i, col=1)

fig.update_layout(
    height=350*len(cryptos),
    title="Bollinger Bands - Retornos semanales de criptomonedas",
    xaxis_title="Fecha",
    yaxis_title="Retorno semanal",
    showlegend=False,
    template="plotly_white"
)
fig.show()

# =============================================================================
# === GRÁFICO ANIMADO DE EVOLUCIÓN DE PRECIOS DE CRIPTOMONEDAS ===
# =============================================================================

P_weeklyC["Date"] = pd.to_datetime(P_weeklyC["Date"], errors="coerce")
P_weeklyC = P_weeklyC.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Detectar rango de fechas
min_date = P_weeklyC["Date"].min()
max_date = P_weeklyC["Date"].max()
print(f"✅ Rango de fechas detectado: {min_date.date()} → {max_date.date()}")

# --- Pasar a formato largo ---
df_long = P_weeklyC.melt(id_vars="Date", var_name="Crypto", value_name="Price")
df_long = df_long.dropna(subset=["Price"]).sort_values(["Crypto", "Date"])
df_long["Date_str"] = df_long["Date"].dt.strftime("%Y-%m-%d")

# --- Construir un dataset acumulativo para cada frame ---
frames = []
for d in sorted(df_long["Date"].unique()):
    temp = df_long[df_long["Date"] <= d].copy()
    temp["Frame"] = pd.to_datetime(d).strftime("%Y-%m-%d")
    frames.append(temp)
df_anim = pd.concat(frames)

# --- Gráfico animado acumulativo ---
fig2 = px.line(
    df_anim,
    x="Date",
    y="Price",
    color="Crypto",
    animation_frame="Frame",
    title="Evolución animada de precios de criptomonedas (semanales)"
)

# --- Ajustes de formato ---
fig2.update_xaxes(
    type="date",
    tickformat="%Y-%m",
    showgrid=True,
    showline=True
)

fig2.update_layout(
    template="plotly_white",
    xaxis_title="Fecha",
    yaxis_title="Precio (USD)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    updatemenus=[{
        "buttons": [
            {"args": [None, {"frame": {"duration": 600, "redraw": True},
                             "fromcurrent": True, "mode": "immediate"}],
             "label": "▶ Reproducir",
             "method": "animate"},
            {"args": [[None], {"frame": {"duration": 0, "redraw": False},
                               "mode": "immediate"}],
             "label": "⏸ Pausar",
             "method": "animate"}
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 60},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }]
)

fig2.show()

# =============================================================================
# === 4️⃣ INDICADORES DE RIESGO Y COMPARACIÓN ===
# =============================================================================

def historical_var(series, level=0.95):
    return np.percentile(series.dropna(), 100*(1-level))

def cvar_historical(series, level=0.95):
    s = series.dropna()
    var = historical_var(s, level)
    tail = s[s <= var]
    return tail.mean() if len(tail) > 0 else var

def sharpe_ratio(series, rf=0.0, periods_per_year=52):
    s = series.dropna()
    if s.std(ddof=1) == 0 or len(s) == 0:
        return np.nan
    ann_ret = s.mean() * periods_per_year
    ann_vol = s.std(ddof=1) * np.sqrt(periods_per_year)
    return (ann_ret - rf) / ann_vol

# Calculamos métricas para cada cripto
metrics_crypto = []
for c in cryptos:
    s = R_weeklyC[c].dropna()
    metrics_crypto.append({
        "Activo": c,
        "Media": s.mean(),
        "Desv.Std": s.std(),
        "Curtosis": kurtosis(s, fisher=False),
        "Skewness": skew(s),
        "VaR 95%": historical_var(s, 0.95),
        "CVaR 95%": cvar_historical(s, 0.95),
        "Sharpe (anual)": sharpe_ratio(s)
    })

df_crypto = pd.DataFrame(metrics_crypto)
print("\n📊 Métricas de riesgo - Criptomonedas:")
print(df_crypto.round(5))


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0",port=10000)
