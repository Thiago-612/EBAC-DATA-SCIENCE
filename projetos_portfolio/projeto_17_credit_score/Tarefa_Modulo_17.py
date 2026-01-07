import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import pointbiserialr
from sklearn.linear_model import LinearRegression
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from translatepy import Translator
from imblearn.over_sampling import SMOTE #imbalanced-learn
import time
import re

pd.set_option('display.width', 120)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('CREDIT_SCORE_P1.csv',
    sep=';',
    encoding='utf-8',
    na_values=['', ' ', 'NA', 'None']
)

print('\n', df.head(20).to_string(), '\n')

print('\nVERIFICAÇÃO DO DATAFRAME ANTES DA TRADUÇÃO: \n')
print(df.info())

PAUSA = 0.03  # pausa leve entre traduções
COLUNAS_SEM_TRADUCAO_TITULO = ["Credit Score"]  # mantém título original

#LIMPEZA LEVE DE STRINGS SEM TRANSFORMAR NAN EM STRING
#sem a transformação os dados ficam viesados, pois nulo vira string e impede a quantificação
df.columns = df.columns.str.strip()
for c in df.select_dtypes(include="object"):
    # aplica strip só em strings; mantém NaN/None intocados
    df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

translator = Translator()

#TRADUZIR OS NOMES DAS COLUNAS EM CAIXA ALTA
colunas_traduzidas = {}
for col in df.columns:
    if col in COLUNAS_SEM_TRADUCAO_TITULO:
        t = col  # mantém título original
    else:
        try:
            t = translator.translate(col, "Portuguese").result
        except Exception:
            t = col  #fallback
    colunas_traduzidas[col] = t.upper()

df.rename(columns=colunas_traduzidas, inplace=True)

#Detectar strings puramente numéricas
def parece_numerico(s: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+([.,]\d+)?", s))

#Coletar valores únicos a traduzir (apenas colunas object, exceto ID)
valores_alvo = set()
for col in df.select_dtypes(include="object").columns:
    unicos = pd.Series(df[col].dropna().unique(), dtype="object")
    for v in unicos:
        if isinstance(v, str):
            vv = v.strip()
            if vv == "" or parece_numerico(vv):
                continue
            valores_alvo.add(vv)
        else:
            # não traduzir nulos/None e não-strings
            continue

#TRADUZIR COM CACHE PULANDO OS NULOS
#Tentei a biblioteca googletrans, mas pela demora excessiva, mesmo tendo internet rápida e máquina boa, tive que cancelar a execução
cache_traducao = {}
for v in valores_alvo:
    try:
        tr = translator.translate(v, "Portuguese").result
        cache_traducao[v] = tr
        time.sleep(PAUSA)
    except Exception:
        cache_traducao[v] = v  #fallback

#FUNÇÃO PARA NAO TRADUZIR NULOS E NAO CONVERTER PARA STRING
def traduzir_seguro(x, cache):
    if pd.isna(x):
        return x
    if not isinstance(x, str):
        return x
    key = x.strip()
    return cache.get(key, x)

# APLICAR TRADUÇÃO E CAIXAS
for col in df.select_dtypes(include="object").columns:

    # aplica tradução sem mexer nos nulos
    df[col] = df[col].apply(lambda x: traduzir_seguro(x, cache_traducao))

df.to_csv("CREDIT_SCORE_P1_TRADUZIDO.csv", index=False, encoding="utf-8-sig")

df_traduzido = pd.read_csv('CREDIT_SCORE_P1_TRADUZIDO.csv',
    encoding='utf-8',
    na_values=['', ' ', 'NA', 'None']
)

print("\n✅ Tradução concluída com sucesso!\n")
print("\nArquivo salvo como 'CREDIT_SCORE_P1_TRADUZIDO.csv'\n")

print('\nDataFrame após tradução: \n')
print('\n', df_traduzido.head(20).to_string(), '\n')
print(df_traduzido.info())

#CONFERENCIA DE VALORES DAS COLUNAS

"""
Análise:
Coluna renda está como object e precisa ser transformada para float.
Pandas só entende padrão de número americano. 20.000,00 vira 20,000.00
Valores das colunas sem valores errados.
Coluna idade possui dados NaN, porém média e mediana estão bem próximos e
não há necessidade de exclusão para preservar a base pequena de dados.
A média será usada nos dados NaN da coluna idade.
"""

colunas = ['IDADE', 'GÊNERO', 'RENDA', 'EDUCAÇÃO', 'ESTADO CIVIL',
    'NÚMERO DE CRIANÇAS', 'PROPRIEDADE DE CASA', 'CREDIT SCORE']

print('\nCONFERINDO OS VALORES INSERIDOS NAS COLUNAS: \n')
for c in colunas:
    print(f"\n📊 Coluna: {c}")
    print(df_traduzido[c].value_counts(dropna=False).to_frame(name='Contagem'))

#Média truncada ou arredondada ? Idade aparece como float utilizando somente a média
# df_traduzido['IDADE_MEDIA'] = df_traduzido['IDADE'].fillna(df_traduzido['IDADE'].mean())
# # media_idade = round(df_traduzido['IDADE'].mean())
# # df_traduzido['IDADE_MEDIA'] = df_traduzido['IDADE'].fillna(media_idade).astype(int)

media_idade = int(df_traduzido['IDADE'].mean())
df_traduzido['IDADE_MEDIA'] = df_traduzido['IDADE'].fillna(media_idade)

#Renda padrão americano
df_traduzido["RENDA"] = (
    df_traduzido["RENDA"]
    .astype(str)
    .str.replace(".", "", regex=False)      # remove milhar
    .str.replace(",", ".", regex=False)     # troca vírgula por ponto
)
df_traduzido["RENDA"] = pd.to_numeric(df_traduzido["RENDA"], errors="coerce")
print('\nCONFERINDO AS ALTERAÇÕES NA COLUNA RENDA: \n')
print(df_traduzido["RENDA"].dtype)
print(df_traduzido["RENDA"].head())

#Análise da média com a mediana(50%)

colunas_numericas = [
    col for col in df_traduzido.select_dtypes(include=['int64', 'float64']).columns
]

# Listas para armazenar resultados
resultados = []

for col in colunas_numericas:
    media = df_traduzido[col].mean()
    mediana = df_traduzido[col].median()

    if pd.notna(media) and pd.notna(mediana):  # ignora colunas vazias
        diff = media - mediana
        resultados.append({
            'Coluna': col,
            'Média': media,
            'Mediana': mediana,
            'Diferença (Média - Mediana)': diff
        })

tabela = pd.DataFrame(resultados)

# Separar por tipo de assimetria
tabela_maior_media = tabela[tabela['Diferença (Média - Mediana)'] > 0].sort_values('Diferença (Média - Mediana)',ascending=False)
tabela_menor_media = tabela[tabela['Diferença (Média - Mediana)'] < 0].sort_values('Diferença (Média - Mediana)',ascending=True)

print('\nCOMPARANDO OS VALORES DA MÉDIA E MEDIANA INSERIDOS NAS COLUNAS DE VARIÁVEIS NUMÉRICAS:')
print('Valores altos na coluna Diferença indicam possíveis outliers')
print("\n📊 Colunas com MÉDIA MAIOR que MEDIANA (cauda à direita):")
print(tabela_maior_media.to_string(index=False))
print("\n📉 Colunas com MÉDIA MENOR que MEDIANA (cauda à esquerda):")
print(tabela_menor_media.to_string(index=False))

#ANÁLISE UNIVARIADA

"""
Análise:
Não há presença de outliers nas variáveis numéricas.
As variáveis categóricas (gênero, educação e estado civil) estão balanceadas.
A variável número de crianças está balanceada se considerarmos grupo de quem tem 
filho e grupo de quem não tem.
A variável propriedade de casa possui o dobro de pessoas com imóvel próprio 
em relação aos que não possuem. Está desbalanceada, mas precisa verificar se 
possui correlação com o score.
Credit Score não está balanceada.
"""

print('\nVERIFICAÇÃO DAS COLUNAS NUMÉRICAS: \n')
print(df_traduzido.describe().to_string(), '\n')

#Conferindo outliers
fig = px.box(
    data_frame=df_traduzido,
    y="RENDA",
    points="all",
    title="Distribuição das Rendas",
    labels={"RENDA": "Renda (R$)"}
)

fig.update_layout(
    yaxis_title="",
    xaxis_title="",
    title_font_size=20,
    template="plotly_white"
)

fig.show()

fig = px.box(
    data_frame=df_traduzido,
    y="IDADE_MEDIA",
    points="all",
    title="Distribuição das Idades",
    labels={"IDADE": "Idade"}
)

fig.update_layout(
    yaxis_title="",
    xaxis_title="",
    title_font_size=20,
    template="plotly_white"
)

fig.show()

categorias = [
    'GÊNERO', 'EDUCAÇÃO', 'ESTADO CIVIL',
    'NÚMERO DE CRIANÇAS', 'PROPRIEDADE DE CASA', 'CREDIT SCORE'
]

fig = make_subplots(
    rows=2, cols=4,
    subplot_titles=categorias,
    specs=[[{'type': 'domain'}] * 4] * 2  # define que todos são gráficos de pizza
)

for i, col in enumerate(categorias):
    row = i // 4 + 1
    col_pos = i % 4 + 1
    contagem = df_traduzido[col].value_counts(dropna=False)

    fig.add_trace(
        go.Pie(
            labels=contagem.index.astype(str),
            values=contagem.values,
            textinfo='percent+label',
            name=col
        ),
        row=row,
        col=col_pos
    )

fig.update_layout(
    height=900,
    width=1200,
    title_text="Distribuição das variáveis categóricas (Balanceamento)",
    showlegend=False,
    template="plotly_white"
)

fig.show()

#ANÁLISE BIVARIADA

"""
Análise:
Percebe-se uma relação entre a idade e o estado civil: Entre 20 e 30 anos predomina o grupo dos solteiros, entre 30 e 40
os resultados estão balanceados, entre 40 e 50 predomina os casados e acima de 50 anos só temos casados. Desta forma pessoas 
mais velhas tendem a ter um casamento.

Rendas maiores são percebidas, na grande maioria, por pessoas mais velhas. Até 69 mil temos os mais jovens, entre 100 mil e 120 mil
temos alguns jovens e após esse valor da renda predomina pessoas com mais de 30 anos.

Na análise do score percebe-se que pessoas com doutorado ou mestrado tem o score mais alto, porém aparenta não ter uma 
relação direta porque pessoas com bacharel e pessoas com ensino médio também possuem score alto. Outros fatores
associados juntos com o nível de escolaridade influenciam no score alto.

Pessoas com a renda maior figuram no grupo de score mais alto, porém em todos os níveis de score existem pessoas com a renda 
mais baixa. Algum fator em conjunto com a renda baixa pode estar elevando o score.

Pessoas com imóvel próprio possuem o score mais alto e a correlação é muito forte. Ter imóvel seria uma forma de garantia ?

Somente uma análise multivariada pode indicar um padrão de clientes com o score mais alto e ajudar na elaboração de
outras perguntas.
"""

# Cria faixas de idade(exemplo: 0–12, 13–24, etc.)
df_traduzido['IDADE_FAIXA'] = pd.cut(
    df_traduzido['IDADE_MEDIA'],
    bins=[20, 30, 40, 50, df_traduzido['IDADE_MEDIA'].max()],
    labels=['20–30', '30–40', '40–50', '50+'],
    include_lowest=True
)

variaveis = [
    ('GÊNERO', 'Gênero'),
    ('ESTADO CIVIL', 'Casado'),
    ('RENDA', 'Renda'),
    ('EDUCAÇÃO', 'Nível escolaridade'),
    ('NÚMERO DE CRIANÇAS', 'Filhos'),
    ('PROPRIEDADE DE CASA', 'Imóvel próprio'),
    ('IDADE_FAIXA', 'Idade do Cliente (faixas)')
]

fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=[f"CREDIT SCORE por {label}" for _, label in variaveis],
    horizontal_spacing=0.10,
    vertical_spacing=0.15
)

for i, (col, label) in enumerate(variaveis):
    row = i // 3 + 1
    col_pos = i % 3 + 1

    # Tabela de frequências: quantos clientes por (SCORE, categoria)
    freq = (
        df_traduzido
        .groupby(['CREDIT SCORE', col])
        .size()
        .reset_index(name='QTDE')
        .dropna(subset=[col])  # evita categorias nulas no gráfico
    )

    # Para cada categoria da variável, cria uma barra separada
    for categoria in freq[col].unique():
        sub = freq[freq[col] == categoria]
        fig.add_trace(
            go.Bar(
                x=sub['CREDIT SCORE'],
                y=sub['QTDE'],
                #name=f"{label}: {categoria}",
                name=str(categoria),  # nome curto na legenda
                legendgroup=label,  # agrupa legendas por variável
                showlegend=(row == 1 and col_pos == 1)  # legenda só no primeiro gráfico
            ),
            row=row,
            col=col_pos
        )

# Layout geral
fig.update_layout(
    title="Distribuição do Score por variáveis",
    title_font_size=22,
    template="plotly_white",
    height=900,
    width=1200,
    barmode='group',
    legend_title_text="Categorias",
    #showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.12,
        xanchor="center",
        x=0.5,
        font=dict(size=11)
    )
)

# Ajustes de eixo
fig.update_xaxes(title_text="Score")
fig.update_yaxes(title_text="Quantidade de clientes")

fig.show()

fig = px.histogram(
    df_traduzido,
    x="RENDA",
    color="IDADE_FAIXA",
    nbins=40,
    opacity=0.75,
    title="Distribuição da Renda por Faixa de Idade",
)

fig.update_layout(
    template="plotly_white",
    bargap=0.05,
    xaxis_title="Renda",
    yaxis_title="Quantidade",
    title_font_size=22,
)

fig.show()

sns.countplot(x='IDADE_FAIXA', hue='ESTADO CIVIL', data=df_traduzido, palette='Set2')
plt.title('Estado Civil por Faixa de Idade')
plt.xlabel('Faixa de idade')
plt.ylabel('Quantidade de clientes')
plt.xticks(rotation=45)

fig = px.bar(
    df_traduzido,
    x="IDADE_FAIXA",
    color="ESTADO CIVIL",
    barmode="group",
    title="Estado Civil por Faixa de Idade",
    labels={
        "IDADE_FAIXA": "Faixa de idade",
        "count": "Quantidade de clientes",
        "ESTADO CIVIL": "Estado civil"
    },
)

fig.update_layout(
    template="plotly_white",
    title_font_size=22,
    xaxis_tickangle=-45,
    legend_title="Estado civil",
)

fig.show()

print('\nConferência do DataFrame antes da codificação: \n')
print('\n', df_traduzido.head(20).to_string(), '\n')
print(df_traduzido.info())

#ANÁLISE MULTIVARIADA COM PARCOODS/PARCATS

"""
Análise:
Parcoords não ficou legal de visualizar.
df_encoded será usado para as bases e futuras análises.
df_traduzido será usado para o parcats.

Pela análise da tabela de correlação entende-se que os valores negativos indicam que enquanto uma variável aumenta a outra 
diminui o valor, como o credit score alto foi codificado como o numeral zero e os outros níveis como 1 e 2 , tal valor
negativo não tem influência na análise dos dados.
As variáveis Idade, Renda e Propriedade de casa possuem as maiores correlações com a variável score, porém isso é uma 
análise bivariada que não explica o perfil de clientes. 
A variável Idade teve sua correlação diminuída com a introdução da média das idades nos valores nulos, pois os nulos não
são utilizados no cálculo da correlação e a média puxa os valores pra baixo. De certa forma, o mais prudente é utilizar a
variável idade_renda para não deixar o perfil de clientes viesados.

A utilização do parcats auxilia muito na análise pela descoberta do perfil de clientes. Tendo a última coluna como score e 
a penúltima como propriedade de imóvel, percebe-se que ter um imóvel é essencial para ter um score alto. A análise parte agora
dessa variável para entender qual o perfil do cliente que tem imóvel próprio. 
Tendo a antepenúltima coluna como idade_faixa percebe-se que pessoas com mais de 30 anos formam a maior parte do grupo de score alto.
Somente uma pequena parcela de pessoas com idade entre 20 e 30 anos tem um score alto: Homens, com renda acima de 80 mil, casados, 
com filhos e mestrados. Nessa faixa de idade podemos ter os critérios para um perfil apto ao score alto.
Ter um imóvel próprio é praticamente uma condição essencial para ter score alto, pois temos esse grupo como score alto em 
todas as categorias de todas as variáveis.
O tamanho pequeno da base de dados dificulta no aprofundamento da análise de perfil. É prudente generalizar utilizando a maioria,
apesar de termos grupos de até cinco pessoas que mostram um padrão diferente. Porém não é prudente generalizar utilizando esses
grupos pequenos. Temos pessoas que não possuem imóvel, mas a renda alta gera um score alto. Outras, mulheres, que possuem imóvel,
filhos e uma renda média possuem score médio. Temos 15 pessoas com score baixo: são mulheres novas, com renda baixa, solteiras,
sem ensino superior e que moram de aluguel.  
"""

#Codificação de variáveis categóricas
df_encoded = df_traduzido.copy()

#Label Encoding para variáveis ordinais
label_cols = ['EDUCAÇÃO', 'IDADE_FAIXA', 'CREDIT SCORE']

for col in label_cols:
    le = LabelEncoder()
    df_encoded[col + '_LE'] = le.fit_transform(df_encoded[col])

#One-Hot Encoding para variáveis nominais (sem ordem)
onehot_cols = ['GÊNERO', 'ESTADO CIVIL', 'PROPRIEDADE DE CASA']

df_encoded = pd.get_dummies(
    df_encoded,
    columns=onehot_cols,
    drop_first=True   # evita multicolinearidade
)

df_encoded.drop(
    ['IDADE', 'EDUCAÇÃO', 'CREDIT SCORE', 'IDADE_FAIXA'],
    axis=1,
    inplace=True
)

#Alterar bool para inteiro
for col in df_encoded.select_dtypes(include='bool').columns:
    df_encoded[col] = df_encoded[col].astype(int)

df_encoded.to_csv('CREDIT_SCORE_P1_ENCODED.csv', index=False)

print('\nVERIFICAÇÃO DO DATAFRAME APÓS CODIFICAÇÃO: \n')
print(df_encoded.info())
print('\n', df_encoded.head(20).to_string(), '\n')

#Parcoods
#Coluna de score codificada
score_col = 'CREDIT SCORE_LE'

#Seleciona todas as colunas numéricas
numeric_cols = df_encoded.select_dtypes(
    include=['int64', 'float64', 'int32', 'float32', 'int8', 'uint8']
).columns.tolist()

#Garante que o score vai ser a ÚLTIMA dimensão
dimensions = [c for c in numeric_cols if c != score_col] + [score_col]

fig = px.parallel_coordinates(
    df_encoded,
    dimensions=dimensions,
    color=score_col,                     # cor baseada no score
    color_continuous_scale="Viridis",
    labels={
        "IDADE": "Idade",
        "RENDA": "Renda (R$)",
        "NÚMERO DE CRIANÇAS": "Filhos",
        "EDUCAÇÃO_LE": "Educação (codificada)",
        "IDADE_FAIXA_LE": "Faixa de idade (codificada)",
        "CREDIT SCORE_LE": "Score de crédito",
        "GÊNERO_Macho": "Gênero = Macho",
        "ESTADO CIVIL_Solteiro": "Solteiro"
    }
)

fig.update_layout(
    title="Parallel Coordinates – Todas as variáveis com Score ao final",
    width=1400,
    height=650,
    template="plotly_white"
)

fig.show()

#Parcats

#faixas de renda
renda_min = df_traduzido['RENDA'].min()
renda_max = df_traduzido['RENDA'].max()

bins = [renda_min - 1, 50000, 80000, 110000, renda_max]
labels = ['≤ 50k', '50k–80k', '80k–110k', '> 110k']

df_traduzido['RENDA_FAIXA'] = pd.cut(
    df_traduzido['RENDA'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

#Dimensões no Parcats
cols = [
    'GÊNERO',
    'IDADE_FAIXA',
    'RENDA_FAIXA',
    'ESTADO CIVIL',
    'EDUCAÇÃO',
    'NÚMERO DE CRIANÇAS',
    'PROPRIEDADE DE CASA',
    'CREDIT SCORE'
]

# Garante que todas essas colunas sejam string (categorias)
for c in cols:
    df_traduzido[c] = df_traduzido[c].astype(str)

# Codifica o CREDIT SCORE só para usar nas cores
score_codes = df_traduzido['CREDIT SCORE'].astype('category').cat.codes

fig = go.Figure(
    go.Parcats(
        dimensions=[{"label": col, "values": df_traduzido[col]} for col in cols],
        line=dict(
            color=score_codes,
            colorscale='Viridis'
        ),
        hoveron='category',
        labelfont=dict(size=14),
        arrangement='freeform'
    )
)

fig.update_layout(
    title="Parallel Categories – Perfil dos clientes por Renda, Idade, Estado Civil e Score",
    font=dict(size=12)
)
fig.show()

#correlação df_traduzido e correlação df_encoded

df_traduzido['SCORE_NUM'] = df_traduzido['CREDIT SCORE'].astype('category').cat.codes
df_encoded['SCORE_NUM'] = df_traduzido['SCORE_NUM']  # garante alinhamento

def correlacao_com_score(df, score_col='SCORE_NUM'):
    y = df[score_col].astype(float)

    num_cols = df.select_dtypes(include=['float64', 'int64', 'int32', 'float32']).columns
    num_cols = [c for c in num_cols if c != score_col]

    resultados = []

    for col in num_cols:
        sub = df[[score_col, col]].dropna()
        # evita colunas constantes
        if sub[col].nunique() <= 1:
            continue
        r = sub[score_col].corr(sub[col])
        resultados.append({'Variável': col, 'Correlação_com_SCORE': r})

    # ordena pelo módulo da correlação
    return (
        pd.DataFrame(resultados)
          .sort_values('Correlação_com_SCORE', key=lambda s: s.abs(), ascending=False)
          .reset_index(drop=True)
    )

corr_trad = correlacao_com_score(df_traduzido)
corr_trad['Fonte'] = 'df_traduzido'

corr_enc = correlacao_com_score(df_encoded)
corr_enc['Fonte'] = 'df_encoded'

tabela_comparacao = pd.concat([corr_trad, corr_enc], ignore_index=True)

print("\n📊 Tabela comparativa de correlação com SCORE:\n")
print(tabela_comparacao.to_string(index=False))

#SEPARAÇÃO BASE DE TREINO E TESTE
"""
Análise:
A variável credit score está desbalanceada acima de 7 vezes entre a maior e menor classe.
"""
x = df_encoded.drop('CREDIT SCORE_LE', axis=1)
y = df_encoded['CREDIT SCORE_LE']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print('\n Tamanho do x_train:', x_train.shape)
print('\n Tamanho do y_train:', y_train.shape)
print('\n Tamanho do x_test:', x_test.shape)
print('\n Tamanho do y_test:', y_test.shape)

#Verificação do balanceamento da variável CREDIT SCORE_LE
score_counts = df_encoded['CREDIT SCORE_LE'].value_counts()
plt.figure(figsize=[10, 10])
score_counts.plot(kind='bar', color=['red', 'green'])
print('\n Porcentagem de cada categoria do Credit Score:')
print('0:Alto 2:Médio 1:Baixo')
print(df_encoded['CREDIT SCORE_LE'].value_counts(normalize=True)*100)

#realizar balanceamento credit score, apenas base de treino

smote = SMOTE(k_neighbors=3, random_state=42)

x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train)

print('\n BALANCEAMENTO:')
print("\nApós SMOTE:")
print("x_train_bal:", x_train_bal.shape)
print("y_train_bal:", y_train_bal.shape)

print("\nDistribuição original do CREDIT SCORE:")
print(y_train.value_counts(normalize=True) * 100)

print("\nDistribuição do CREDIT STORE após SMOTE:")
print(y_train_bal.value_counts(normalize=True) * 100)

x_train_bal.to_csv("x_train_bal.csv", index=False)
y_train_bal.to_csv("y_train_bal.csv", index=False)
x_test.to_csv("x_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
