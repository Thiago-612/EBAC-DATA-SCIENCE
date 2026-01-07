import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


data = {
    'Nome': ['Alice', 'Joao', 'Charlie', 'David', 'Eva', 'Diego', 'Denize', 'Claudio'],
    'Idade': [25, 30, 35, 40, 45, 60, 22, 24],
    'Profissão': ['Engenheiro', 'Médico', 'Professor', 'Advogado', 'Médico','Engenheiro', 'Estudante','Estudante'],
    'Salário': ['4500', '8000', '5000', '10000', '12000','15000', '1200','1500'],
    'Limite_Credito': ['2500', '4000', '4000', '1000', '10000','2000', '500','250'],
    'Historico_Inadimplencia': ['0', '0', '0', '1', '0','1', '0','1'],
    'Estado_Civil': ['Casamento', 'Casamento', 'Solteiro', 'Solteiro', 'Casamento','Solteiro', 'Solteiro','Solteiro'],
    'Imovel_Proprio': ['0', '0', '0', '1', '1','1', '0','0']
}
df = pd.DataFrame(data)

#DIAGRAMA DE CATEGORIAS PARALELAS PARA VISUALIZAR MELHOR AS RELAÇÕES ENTRE OS CONJUNTOS

# Conversões de tipo (strings -> numéricos onde faz sentido)
for col in ['Salário', 'Limite_Credito', 'Historico_Inadimplencia', 'Imovel_Proprio']:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

# Mapear Estado_Civil para uma ordem fixa nas dimensões categóricas
ordem_estado_civil = ['Solteiro', 'Casamento']

# Definição das dimensões do Parcats
salario_dim = dict(
    values=df['Salário'],
    label="Salário",
    categoryorder='category ascending'  # ordena categorias (se fossem categóricas)
)

imovel_dim = dict(
    values=df['Imovel_Proprio'],
    label="Imóvel",
    categoryarray=[0, 1],              # valores presentes nos dados (inteiros 0/1)
    ticktext=['Não possui', 'Possui']  # rótulos que aparecem na dimensão
)

estado_civil_dim = dict(
    values=df['Estado_Civil'],
    label="Estado Civil",
    categoryarray=ordem_estado_civil   # força a ordem das categorias
)

inadimplencia_dim = dict(
    values=df['Historico_Inadimplencia'],
    label="Inadimplência",
    categoryarray=[0, 1],
    ticktext=['Nunca tiveram', 'Tiveram']
)

limite_dim = dict(
    values=df['Limite_Credito'],
    label="Limite de Crédito",
    categoryorder='category ascending'
)

# Cor da linha por limite de crédito (numérico)
color = df['Limite_Credito'].astype(float)

# Gradiente customizado: 0% vermelho escuro → 50% vermelho claro → 100% azul
red_to_blue = [
    [0.0,  '#b30000'],  # vermelho escuro
    [0.20, '#e34a33'],
    [0.40, '#74add1'],
    [1.0,  '#2b8cbe']   # azul
]

# Criação do gráfico Parcats
fig = go.Figure(
    data=[
        go.Parcats(
            dimensions=[salario_dim, imovel_dim, estado_civil_dim, inadimplencia_dim, limite_dim],
            line=dict(color=color, colorscale=red_to_blue, cmin=float(color.min()), cmax=float(color.max())),
            hoveron='color',
            hoverinfo='count+probability',
            labelfont=dict(size=14),
            tickfont=dict(size=12),
            arrangement='freeform',
            bundlecolors=True
        )
    ]
)

fig.update_layout(margin=dict(l=40, r=40, t=40, b=40), title="Fluxo por Salário, Imóvel, Estado Civil, Inadimplência e Limite")
fig.show()

#QUAIS FATORES TÊM UMA CORRELAÇÃO MAIOR COM O LIMITE
import plotly.figure_factory as ff

corr = df[['Salário', 'Idade', 'Limite_Credito', 'Historico_Inadimplencia', 'Imovel_Proprio']].corr()#, 'Estado_Civil']].corr()
fig = ff.create_annotated_heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.columns.tolist(),
    colorscale='RdBu',
    showscale=True
)
fig.update_layout(title="Mapa de correlação entre variáveis")
fig.show()


# fig = px.treemap(
#     df,
#     path=['Salário', 'Idade', 'Historico_Inadimplencia', 'Imovel_Proprio', 'Estado_Civil'],
#     values='Limite_Credito',
#     color='Limite_Credito',
#     color_continuous_scale='RdBu'
# )
# fig.update_layout(title="Segmentação hierárquica dos fatores de limite")
# fig.show()

#VERIFICAR SE QUEM GANHA MAIS TEM MAIS LIMITE
#INADIMPLENTES TEM LIMITE MENOR ?
#IDADE INFLUENCIA ?
fig = px.scatter(
    df,
    x='Salário',
    y='Limite_Credito',
    color='Historico_Inadimplencia',
    size='Idade',
    hover_data=['Nome', 'Profissão'],
    trendline='ols'  # linha de regressão automática
)
fig.update_layout(title="Relação entre salário e limite de crédito")
fig.show()
