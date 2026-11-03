"""
app.py - ETAPA 7: A PLATAFORMA WEB (STREAMLIT)
==============================================

Rodar:  streamlit run app.py   (abre no navegador em http://localhost:8501)

COMO O STREAMLIT FUNCIONA
  Voce escreve um script Python comum e cada comando "st.algo(...)" vira um elemento
  na pagina: st.title -> titulo, st.dataframe -> tabela, st.line_chart -> grafico,
  st.selectbox -> lista de escolha. Toda vez que o usuario mexe em algo (escolhe um
  ativo, por exemplo), o Streamlit roda o script de novo de cima para baixo com o
  novo valor. Nao e preciso escrever HTML nem JavaScript.

ARQUITETURA
  O site NAO baixa dados nem treina modelos: so LE os arquivos gerados pelos scripts
  01 a 03. Assim ele abre rapido. Para atualizar as previsoes, roda-se
  "python atualizar.py --hoje" (manualmente ou agendado) e o site mostra os novos arquivos.
"""
import pandas as pd
import streamlit as st

import config

# Titulo da aba do navegador e layout largo
st.set_page_config(page_title="Previsão de Movimentos de Ativos", layout="wide")

# Nomes para exibir (o codigo usa "logistica", a tela mostra "Regressão Logística")
NOME_CONJ = {"tecnicos": "Indicadores técnicos", "ampliado": "Técnicos + risco e retorno"}
_MOD = {"logistica": "Regressão Logística", "random_forest": "Random Forest",
        "rede_neural": "Rede Neural (MLP)"}
NOME_MOD = {m: _MOD[m] for m in config.MODELOS}
NOME_TIPO = {"etf": "ETF", "acao": "Ação", "fii": "FII"}


# @st.cache_data: o Streamlit guarda o resultado desta funcao na memoria. Como o
# script roda de novo a cada clique, sem o cache os arquivos seriam relidos toda vez.
@st.cache_data
def carregar():
    """Le todos os arquivos de resultados num dicionario D."""
    r, p = config.RESULTADOS, config.DADOS_PROCESSADOS
    d = {
        "dados": pd.read_parquet(p / "dataset.parquet"),
        "ativos": pd.read_csv(p / "ativos.csv"),
        "resumo": pd.read_csv(r / "metricas_resumo.csv"),
        "por_ativo": pd.read_csv(r / "metricas_por_ativo.csv"),
        "por_ano": pd.read_csv(r / "metricas_por_ano.csv"),
        "comparacao": pd.read_csv(r / "comparacao_modelos.csv"),
        "estabilidade": pd.read_csv(r / "estabilidade.csv"),
        "confusao": pd.read_csv(r / "matrizes_confusao.csv"),
        "backtest": pd.read_csv(r / "backtest_por_ativo.csv"),
        "curvas": pd.read_parquet(r / "curvas_backtest.parquet"),
        "importancia": pd.read_csv(r / "importancia_variaveis.csv"),
        "atual": pd.read_csv(r / "previsao_atual.csv"),
        "previsoes": pd.read_parquet(r / "previsoes.parquet"),
    }
    arq = config.DADOS_BRUTOS / "ultima_atualizacao.txt"
    d["atualizado"] = arq.read_text(encoding="utf-8") if arq.exists() else "desconhecida"
    return d


# Se algum arquivo ainda nao existir (pipeline nunca rodou), mostra uma mensagem clara
try:
    D = carregar()
except FileNotFoundError as erro:
    st.error(f"Arquivo não encontrado: {erro.filename}. Rode antes  python atualizar.py")
    st.stop()

st.title("Previsão de Movimentos de Ativos Financeiros com Machine Learning")
st.caption(f"TCC — Ciência da Computação (UVA) · Gustavo Pereira da Silva · "
           f"Dados atualizados em {D['atualizado']} · "
           "Material educacional: não é recomendação de investimento.")

# Barra lateral: escolha do conjunto de variaveis (so aparece mais de uma opcao
# se o conjunto "ampliado" estiver ligado no config.py)
with st.sidebar:
    conj = st.radio("Variáveis", list(config.CONJUNTOS), format_func=NOME_CONJ.get,
                    index=len(config.CONJUNTOS) - 1)
    st.caption("Variáveis: variação do VIX e nove indicadores técnicos (CCI, MACD normalizado/PPO, "
               "Williams %R, estocástico (SMI), TRIX, Bollinger %B, CMO, DPO e ROC).")

# A pagina tem 4 abas; cada bloco "with aba_x:" desenha o conteudo de uma delas
aba_hoje, aba_ativo, aba_comp, aba_metodo = st.tabs(
    ["Previsões de hoje", "Ativo", "Comparação dos modelos", "Método"])

# ---------------------------------------------------------------------------
# ABA 1: a previsao do proximo pregao para todos os ativos + acerto recente
with aba_hoje:
    at = D["atual"].merge(D["ativos"][["ativo", "nome", "tipo"]], on="ativo")
    st.subheader(f"Probabilidade de alta no próximo pregão (após {at['data_referencia'].max()})")
    tab = pd.DataFrame({"Ativo": at["ativo"], "Nome": at["nome"],
                        "Tipo": at["tipo"].map(NOME_TIPO)})
    for m, nome in NOME_MOD.items():
        tab[nome] = at[f"prob_alta_{conj}__{m}"]
    n_mod = len(NOME_MOD)
    votos = (tab[list(NOME_MOD.values())] >= 0.5).sum(axis=1)
    tab["Consenso"] = [f"alta ({v}/{n_mod})" if v > n_mod / 2 else
                       f"baixa ({n_mod - v}/{n_mod})" if v < n_mod / 2 else "dividido"
                       for v in votos]
    st.dataframe(tab.style.format({n: "{:.1%}" for n in NOME_MOD.values()})
                 .background_gradient(subset=list(NOME_MOD.values()), cmap="RdBu", vmin=0.3, vmax=0.7),
                 width="stretch", hide_index=True)
    st.caption("A probabilidade vem de modelos treinados com todo o histórico disponível. "
               "Valores perto de 50% indicam pouca convicção.")

    st.subheader("Acerto recente (todos os ativos)")
    pv = D["previsoes"].copy()
    pv["data"] = pd.to_datetime(pv["data"])
    fim = pv["data"].max()
    linhas = []
    for rot, dias in (("Última semana", 7), ("Último mês", 31), ("Último trimestre", 92)):
        g = pv[pv["data"] > fim - pd.Timedelta(days=dias)]
        lin = {"Janela": rot, "Previsões": len(g)}
        for m, nome in NOME_MOD.items():
            lin[nome] = (g[f"pred_{conj}__{m}"] == g["alvo"]).mean()
        lin["Sempre alta"] = (g["alvo"] == 1).mean()
        linhas.append(lin)
    cols_pct = list(NOME_MOD.values()) + ["Sempre alta"]
    st.dataframe(pd.DataFrame(linhas).style.format({c: "{:.1%}" for c in cols_pct}),
                 width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# ABA 2: um ativo escolhido: previsao, graficos, desempenho e backtest
with aba_ativo:
    incl = D["ativos"][D["ativos"]["incluido"]]
    rotulos = {f"{r.ativo} — {r.nome} ({NOME_TIPO[r.tipo]})": r.ativo for r in incl.itertuples()}
    escolha = rotulos[st.selectbox("Ativo", list(rotulos))]
    df = D["dados"][D["dados"]["ativo"] == escolha].set_index("data")

    a = D["atual"][D["atual"]["ativo"] == escolha]
    if not a.empty:
        st.subheader(f"Previsão para o pregão seguinte a {a['data_referencia'].iloc[0]}")
        for col, (m, nome) in zip(st.columns(len(NOME_MOD)), NOME_MOD.items()):
            p = a[f"prob_alta_{conj}__{m}"].iloc[0]
            col.metric(nome, "ALTA" if p >= 0.5 else "BAIXA", f"prob. de alta {p:.1%}",
                       delta_color="off")

    st.subheader("Preço e média móvel de 200 pregões")
    st.line_chart(pd.DataFrame({"Preço": df["preco"],
                                "Média 200": df["preco"] / (1 + df["dist_mm200"])}))
    indicador = st.selectbox("Indicador", config.TODAS_VARIAVEIS, index=1)
    st.line_chart(df[indicador])

    st.subheader("Desempenho fora da amostra")
    met = D["por_ativo"][(D["por_ativo"]["ativo"] == escolha)
                         & D["por_ativo"]["conjunto"].isin([conj, "-"])]
    met = met.drop(columns=["ativo", "tipo", "conjunto"]).set_index("modelo")
    st.dataframe(met.style.format({c: "{:.3f}" for c in met.columns if c != "n"}), width="stretch")

    st.subheader("Backtest: no ativo quando o modelo prevê alta, senão CDI")
    curva = D["curvas"][D["curvas"]["ativo"] == escolha].drop(columns="ativo").set_index("data")
    manter = ["Comprar e manter", "CDI"] + [c for c in curva.columns if f"[{conj}]" in c]
    st.line_chart(curva[manter])
    bt = D["backtest"][(D["backtest"]["ativo"] == escolha)
                       & D["backtest"]["estrategia"].isin(manter)]
    st.dataframe(bt.drop(columns=["ativo", "tipo"]).set_index(["periodo", "estrategia"]).style.format({
        "retorno_total": "{:.1%}", "retorno_anual": "{:.1%}", "retorno_anual_sem_custo": "{:.1%}",
        "volatilidade_anual": "{:.1%}",
        "sharpe": "{:.2f}", "drawdown_maximo": "{:.1%}", "tempo_no_ativo": "{:.0%}"}),
        width="stretch")
    st.caption(f"Custo de {config.CUSTO_POR_TROCA:.2%} por troca. Execução suposta no fechamento "
               "do dia da previsão; na prática seria no pregão seguinte.")

# ---------------------------------------------------------------------------
# ABA 3: comparacao dos modelos (a pergunta de pesquisa)
with aba_comp:
    def tabela(seg):
        r = D["resumo"][(D["resumo"]["segmento"] == seg)
                        & D["resumo"]["conjunto"].isin([conj, "-"])]
        return r.pivot(index="modelo", columns="valor", values="acuracia")

    st.subheader("Acurácia: período de validação x período reservado")
    st.dataframe(tabela("periodo").style.format("{:.2%}"), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Por tipo de ativo")
        st.dataframe(tabela("tipo").style.format("{:.2%}"), width="stretch")
    with c2:
        st.subheader("Por regime de volatilidade")
        st.dataframe(tabela("regime_vol").style.format("{:.2%}"), width="stretch")

    st.subheader("Robustez no tempo: acurácia por ano")
    pa = D["por_ano"][D["por_ano"]["conjunto"].isin([conj, "-"])]
    st.line_chart(pa.pivot(index="ano", columns="modelo", values="acuracia"))

    st.subheader("A Random Forest supera a Regressão Logística? (teste de McNemar)")
    cp = D["comparacao"]
    cp = cp[(cp["conjunto"] == conj) & (cp["modelo_a"] == "Regressao Logistica")
            & (cp["modelo_b"] == "Random Forest")]
    st.dataframe(cp.drop(columns=["conjunto", "modelo_a", "modelo_b"]).style.format(
        {"acuracia_a": "{:.2%}", "acuracia_b": "{:.2%}", "p_valor_mcnemar": "{:.4f}"}),
        width="stretch", hide_index=True)
    st.caption("acuracia_a = Logística, acuracia_b = Random Forest. p < 0,05 indica diferença "
               "significativa entre os dois modelos naquele recorte.")
    st.dataframe(D["estabilidade"][D["estabilidade"]["conjunto"] == conj], width="stretch",
                 hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Matrizes de confusão")
        st.dataframe(D["confusao"][D["confusao"]["conjunto"] == conj].drop(columns="conjunto"),
                     width="stretch", hide_index=True)
    with c2:
        st.subheader("Importância das variáveis (Random Forest)")
        imp = D["importancia"][D["importancia"]["conjunto"] == conj]
        st.bar_chart(imp.set_index("variavel")["importancia_rf"].sort_values())

# ---------------------------------------------------------------------------
# ABA 4: explicacao do metodo, em texto (st.markdown entende **negrito** etc.)
with aba_metodo:
    st.markdown(f"""
**Problema.** Classificar se o retorno do próximo pregão (D+{config.HORIZONTE}) será positivo
(alta) ou não (baixa): um problema de classificação binária supervisionada.

**Amostra.** ETFs, ações e FIIs da B3 com histórico de {config.DATA_INICIO} a {config.DATA_FIM}
e volume financeiro diário mediano de pelo menos R$ {config.LIQUIDEZ_MINIMA:,.0f}.
Dados: Yahoo Finance (preços ajustados), VIX e CDI (Banco Central, SGS 12).

**Variáveis.** Variação do VIX e nove indicadores técnicos clássicos (CCI, MACD normalizado (PPO), Williams %R, estocástico,
TRIX, Bollinger, CMO, DPO e ROC){" e, no conjunto ampliado, métricas de risco e retorno" if "ampliado" in config.CONJUNTOS else ""}.
Todas usam apenas informação disponível no fechamento do dia da previsão; o VIX entra com
um dia de defasagem, porque o mercado americano fecha depois da B3.

**Modelos.** {", ".join(NOME_MOD.values())}, com hiperparâmetros fixados antes dos testes.

**Validação.** Walk-forward com janela expansiva: primeiro treino de {config.TREINO_INICIAL}
pregões, testes sucessivos de {config.JANELA_TESTE}. O período a partir de
{config.INICIO_RESERVADO} é reservado e reportado separadamente.

**Comparações.** Linhas de base (sempre alta, classe majoritária, persistência), teste
binomial contra "sempre alta" e teste de McNemar entre modelos, no geral, por ano e por
regime de volatilidade.
""")
