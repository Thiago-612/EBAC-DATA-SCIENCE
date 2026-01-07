# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, callback_context

# =========================================================
# CONFIG: caminho do seu CSV (ou deixe None p/ exemplo)
CSV_PATH = "ecommerce_preparados_Tarefa.csv"  # troque para seu arquivo
# =========================================================

# ---------- Carrega dados ----------
if CSV_PATH and os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    # fallback: dataset de exemplo
    df = px.data.tips()
    # renomeia p/ PT-BR só pra parecer com seu estilo
    df = df.rename(columns={
        "total_bill": "Valor_Total",
        "sex": "Sexo",
        "smoker": "Fumante",
        "day": "Dia_Semana",
        "time": "Periodo",
        "size": "Tamanho_Grupo"
    })

# ---------- Limpezas leves ----------
# remove colunas 100% vazias/duplicadas
df = df.loc[:, ~df.columns.duplicated()]
df = df.dropna(axis=1, how="all")

# tenta converter strings "numéricas" em número sem quebrar
def try_to_numeric(s: pd.Series) -> pd.Series:
    if s.dtype == "object":
        # troca vírgula por ponto se necessário
        s2 = s.str.replace(",", ".", regex=False)
        s_num = pd.to_numeric(s2, errors="ignore")
        return s_num
    return s

for c in df.columns:
    df[c] = try_to_numeric(df[c])

# --------- Detecta tipos automaticamente ---------
# numéricas: dtypes numéricos
num_cols = df.select_dtypes(include="number").columns.tolist()
# categóricas: não-numéricas + numéricas com baixa cardinalidade (<= 12)
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
low_card_num = [c for c in num_cols if df[c].nunique(dropna=True) <= 12]
cat_cols = sorted(list(set(cat_cols + low_card_num)))
# garante pelo menos 1 coluna em cada grupo
if not num_cols:
    # cria uma numérica fake se necessário
    df["_contagem"] = 1
    num_cols = ["_contagem"]
if not cat_cols:
    # cria uma categórica fake se necessário
    df["_categoria"] = "categoria"
    cat_cols = ["_categoria"]

# opções p/ dropdowns
def opts(cols): return [{"label": c, "value": c} for c in cols]

# ================== APP DASH ==================
app = Dash(__name__)
app.title = "Dashboard Genérico - Plotly/Dash"

app.layout = html.Div([
    html.H2("Dashboard Genérico para Análise Exploratória", style={"margin":"10px 0"}),

    html.Div([
        html.Div([
            html.Label("Coluna Numérica (X)"),
            dcc.Dropdown(id="x-num", options=opts(num_cols), value=num_cols[0], clearable=False),
        ], style={"width":"24%", "display":"inline-block", "padding":"0 8px"}),

        html.Div([
            html.Label("Coluna Numérica (Y)"),
            dcc.Dropdown(id="y-num", options=opts(num_cols), value=(num_cols[1] if len(num_cols)>1 else num_cols[0]), clearable=False),
        ], style={"width":"24%", "display":"inline-block", "padding":"0 8px"}),

        html.Div([
            html.Label("Coluna Categórica (Cor/Hue)"),
            dcc.Dropdown(id="cat-color", options=opts(cat_cols), value=(cat_cols[0] if cat_cols else None), clearable=True),
        ], style={"width":"24%", "display":"inline-block", "padding":"0 8px"}),

        html.Div([
            html.Label("Outra Categórica (Faceta col)"),
            dcc.Dropdown(id="cat-facet", options=opts(cat_cols), value=None, clearable=True),
        ], style={"width":"24%", "display":"inline-block", "padding":"0 8px"}),
    ], style={"margin":"10px 0"}),

    dcc.Tabs([
        dcc.Tab(label="Visão Geral", children=[
            html.Br(),
            html.Div(id="cards-overview", style={"display":"flex","gap":"16px","flexWrap":"wrap"}),
            html.Br(),
            html.Div([
                html.Div([dcc.Graph(id="hist-overview")], style={"width":"50%","display":"inline-block","padding":"0 8px"}),
                html.Div([dcc.Graph(id="bar-overview")],  style={"width":"50%","display":"inline-block","padding":"0 8px"}),
            ])
        ]),

        dcc.Tab(label="Dispersão + Regressão", children=[
            html.Br(),
            dcc.Graph(id="scatter-reg")
        ]),

        dcc.Tab(label="Box / Violin", children=[
            html.Br(),
            html.Div([
                html.Div([
                    html.Label("Categórica (eixo X)"),
                    dcc.Dropdown(id="x-cat-box", options=opts(cat_cols), value=(cat_cols[0] if cat_cols else None), clearable=False),
                ], style={"width":"30%","display":"inline-block","padding":"0 8px"}),
                html.Div([
                    html.Label("Numérica (eixo Y)"),
                    dcc.Dropdown(id="y-num-box", options=opts(num_cols), value=num_cols[0], clearable=False),
                ], style={"width":"30%","display":"inline-block","padding":"0 8px"}),
                html.Div([
                    html.Label("Tipo de gráfico"),
                    dcc.Dropdown(
                        id="kind-box",
                        options=[{"label":"Boxplot","value":"box"},{"label":"Violin","value":"violin"}],
                        value="box",
                        clearable=False
                    ),
                ], style={"width":"30%","display":"inline-block","padding":"0 8px"}),
            ]),
            dcc.Graph(id="box-graph")
        ]),

        dcc.Tab(label="Histograma / KDE", children=[
            html.Br(),
            html.Div([
                html.Div([
                    html.Label("Numérica p/ Histograma"),
                    dcc.Dropdown(id="num-hist", options=opts(num_cols), value=num_cols[0], clearable=False),
                ], style={"width":"30%","display":"inline-block","padding":"0 8px"}),
                html.Div([
                    html.Label("Categórica (cor)"),
                    dcc.Dropdown(id="cat-hist", options=opts(cat_cols), value=None, clearable=True),
                ], style={"width":"30%","display":"inline-block","padding":"0 8px"}),
                html.Div([
                    html.Label("Bins"),
                    dcc.Slider(id="bins-hist", min=10, max=100, step=5, value=40, marks=None, tooltip={"placement":"bottom","always_visible":True})
                ], style={"width":"40%","display":"inline-block","padding":"10px 8px 0"}),
            ]),
            dcc.Graph(id="hist-graph")
        ]),

        dcc.Tab(label="Correlação (Heatmap)", children=[
            html.Br(),
            dcc.Graph(id="corr-heat")
        ]),

        dcc.Tab(label="Treemap / Sunburst", children=[
            html.Br(),
            html.Div([
                html.Div([
                    html.Label("Path (até 3 categóricas)"),
                    dcc.Dropdown(id="path-treemap", options=opts(cat_cols), value=cat_cols[:2], multi=True),
                ], style={"width":"49%","display":"inline-block","padding":"0 8px"}),
                html.Div([
                    html.Label("Valor numérico (soma)"),
                    dcc.Dropdown(id="value-treemap", options=opts(num_cols), value=num_cols[0], clearable=False),
                ], style={"width":"49%","display":"inline-block","padding":"0 8px"}),
            ]),
            html.Div([
                html.Div([dcc.Graph(id="treemap")],  style={"width":"50%","display":"inline-block","padding":"0 8px"}),
                html.Div([dcc.Graph(id="sunburst")], style={"width":"50%","display":"inline-block","padding":"0 8px"}),
            ])
        ]),

        dcc.Tab(label="Parallel Categories (opcional)", children=[
            html.Br(),
            dcc.Markdown("Use até 4 colunas **categóricas** para o Parcats:"),
            dcc.Dropdown(id="parcats-cols", options=opts(cat_cols), value=cat_cols[:4], multi=True),
            dcc.Graph(id="parcats-graph")
        ]),
    ])
], style={"maxWidth":"1400px","margin":"0 auto","padding":"10px"})

# ================== CALLBACKS ==================

@app.callback(
    Output("cards-overview","children"),
    Input("x-num","value"), Input("y-num","value")
)
def cards_overview(xnum, ynum):
    nlin = len(df)
    ncol = len(df.columns)
    nmiss = int(df.isna().sum().sum())
    cards = []
    style = {"border":"1px solid #ddd","borderRadius":"8px","padding":"12px","minWidth":"180px","background":"#fafafa"}
    cards.append(html.Div([html.H4("Linhas"), html.H3(f"{nlin:,}".replace(",","."))], style=style))
    cards.append(html.Div([html.H4("Colunas"), html.H3(f"{ncol:,}".replace(",","."))], style=style))
    cards.append(html.Div([html.H4("Valores ausentes"), html.H3(f"{nmiss:,}".replace(",","."))], style=style))
    cards.append(html.Div([html.H4("Numéricas"), html.P(", ".join(num_cols[:6]) + ("..." if len(num_cols)>6 else ""))], style=style))
    cards.append(html.Div([html.H4("Categóricas"), html.P(", ".join(cat_cols[:6]) + ("..." if len(cat_cols)>6 else ""))], style=style))
    return cards

@app.callback(
    Output("hist-overview","figure"),
    Output("bar-overview","figure"),
    Input("x-num","value"), Input("cat-color","value")
)
def overview_figs(numeric_col, cat):
    fig1 = px.histogram(df, x=numeric_col, nbins=40, marginal="box", title=f"Histograma de {numeric_col}")
    fig1.update_layout(bargap=0.05)

    top_cat = None
    if cat:
        vc = df[cat].astype(str).value_counts().reset_index()
        vc.columns = [cat, "Contagem"]
        fig2 = px.bar(vc, x=cat, y="Contagem", title=f"Top categorias em {cat}")
    else:
        # sem categórica, mostra a de maior cardinalidade baixa
        base = cat_cols[0]
        vc = df[base].astype(str).value_counts().reset_index()
        vc.columns = [base, "Contagem"]
        fig2 = px.bar(vc, x=base, y="Contagem", title=f"Top categorias em {base}")
    fig2.update_layout(xaxis_tickangle=0)
    return fig1, fig2

@app.callback(
    Output("scatter-reg","figure"),
    Input("x-num","value"), Input("y-num","value"),
    Input("cat-color","value"), Input("cat-facet","value")
)
def scatter_reg(xnum, ynum, hue, facet):
    df2 = df[[xnum, ynum] + ([hue] if hue else []) + ([facet] if facet else [])].dropna()
    fig = px.scatter(df2, x=xnum, y=ynum, color=hue, facet_col=facet,
                     trendline="ols", opacity=0.7, hover_data=df.columns)
    fig.update_layout(title=f"Dispersão e Regressão — {xnum} vs {ynum}")
    return fig

@app.callback(
    Output("box-graph","figure"),
    Input("x-cat-box","value"), Input("y-num-box","value"), Input("cat-color","value"),
    Input("kind-box","value")
)
def box_violin(xcat, ynum, hue, kind):
    df2 = df[[xcat, ynum] + ([hue] if hue else [])].dropna()
    if kind == "box":
        fig = px.box(df2, x=xcat, y=ynum, color=hue, points="outliers", title=f"Boxplot — {ynum} por {xcat}")
    else:
        fig = px.violin(df2, x=xcat, y=ynum, color=hue, box=True, points="outliers", title=f"Violin — {ynum} por {xcat}")
    fig.update_layout(xaxis_tickangle=0)
    return fig

@app.callback(
    Output("hist-graph","figure"),
    Input("num-hist","value"), Input("cat-hist","value"), Input("bins-hist","value")
)
def histo(num_col, cat_col, bins):
    df2 = df[[num_col] + ([cat_col] if cat_col else [])].dropna()
    fig = px.histogram(df2, x=num_col, color=cat_col, nbins=int(bins), barmode="overlay",
                       marginal="rug", opacity=0.75, title=f"Histograma — {num_col}")
    return fig

@app.callback(
    Output("corr-heat","figure"),
    Input("x-num","value")
)
def corr_heat(_):
    if len(num_cols) < 2:
        return go.Figure().update_layout(title="Necessário ao menos 2 colunas numéricas.")
    corr = df[num_cols].corr()
    fig = ff.create_annotated_heatmap(
        z=np.round(corr.values, 2),
        x=num_cols, y=num_cols,
        colorscale="RdBu", showscale=True, reversescale=True
    )
    fig.update_layout(title="Mapa de Correlação (numéricas)")
    return fig

@app.callback(
    Output("treemap","figure"),
    Output("sunburst","figure"),
    Input("path-treemap","value"), Input("value-treemap","value")
)
def tree_sun(path_cols, val_col):
    if not path_cols: path_cols = [cat_cols[0]]
    df2 = df[path_cols + [val_col]].dropna()
    fig1 = px.treemap(df2, path=path_cols, values=val_col, color=val_col, color_continuous_scale="Blues",
                      title="Treemap")
    fig2 = px.sunburst(df2, path=path_cols, values=val_col, color=val_col, color_continuous_scale="Blues",
                       title="Sunburst")
    return fig1, fig2

@app.callback(
    Output("parcats-graph","figure"),
    Input("parcats-cols","value")
)
def parcats(cols):
    if not cols or len(cols) < 2:
        return go.Figure().update_layout(title="Selecione 2 a 4 colunas categóricas")
    cols = cols[:4]
    df2 = df[cols].dropna()
    dims = [dict(values=df2[c].astype(str), label=c) for c in cols]
    fig = go.Figure(
        data=[go.Parcats(
            dimensions=dims,
            line=dict(color=np.arange(len(df2)), colorscale="Tealrose", shape="hspline"),
            arrangement="freeform",
            bundlecolors=True,
            hoveron="color", hoverinfo="count+probability"
        )]
    )
    fig.update_layout(title="Parallel Categories (Parcats)")
    return fig

# ================== RUN ==================
if __name__ == "__main__":
    # modo dev
    app.run_server(debug=True, port=8050)
