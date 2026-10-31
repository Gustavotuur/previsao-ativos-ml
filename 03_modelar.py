"""
03_modelar.py - ETAPAS 5 E 6: MODELAGEM DE MACHINE LEARNING E ANALISE DOS RESULTADOS
=====================================================================================

PERGUNTA DE PESQUISA
  A Random Forest supera a Regressao Logistica na previsao da direcao do proximo
  pregao (D+1)? E essa vantagem se mantem ao longo dos anos, em mercados calmos e
  agitados e nos diferentes tipos de ativo?

O QUE ESTE SCRIPT FAZ
  1. Para cada ativo, roda a validacao WALK-FORWARD: treina os modelos no passado e
     preve o periodo seguinte, avancando no tempo ate o fim da amostra.
  2. Compara os modelos com LINHAS DE BASE (estrategias "burras"), porque so faz
     sentido comemorar se o modelo acertar mais do que um chute bem informado:
       - sempre alta:          preve sempre "sobe"
       - classe majoritaria:   preve sempre o que foi mais frequente no treino
       - persistencia:         preve que amanha sera igual a hoje (subiu -> sobe)
  3. Calcula metricas e testes estatisticos por segmento (geral, periodo, ano,
     tipo de ativo, regime de volatilidade).
  4. Simula um BACKTEST: ficar no ativo quando o modelo preve alta e no CDI quando
     preve baixa, pagando 0,10% a cada troca.
  5. Treina os modelos com todo o historico e gera a previsao para o proximo pregao
     (usada no site).

Os hiperparametros dos modelos sao FIXOS (definidos antes de ver qualquer resultado),
para nao haver "ajuste ao teste".

Arquivos gerados em resultados/:
  previsoes.parquet            a previsao de cada modelo em cada dia de teste
  metricas_resumo.csv          metricas gerais e por periodo, tipo de ativo e regime
  metricas_por_ativo.csv       metricas de cada ativo
  metricas_por_ano.csv         metricas por ano (robustez no tempo)
  comparacao_modelos.csv       teste de McNemar entre os modelos
  estabilidade.csv             em quantos anos/ativos a Random Forest superou a Logistica
  matrizes_confusao.csv        matriz de confusao de cada modelo
  backtest_por_ativo.csv       resultado financeiro simulado
  curvas_backtest.parquet      patrimonio acumulado dia a dia (para graficos)
  backtest_sensibilidade.csv   retorno do backtest com diferentes custos por troca
  backtest_aleatorio.csv       modelo comparado com estrategias aleatorias de mesma exposicao
  importancia_variaveis.csv    quais indicadores cada modelo mais usou
  previsao_atual.csv           previsao para o proximo pregao
"""
import time
import warnings
from itertools import combinations   # gera todos os pares possiveis de uma lista

import numpy as np
import pandas as pd
from joblib import Parallel, delayed  # roda varios ativos ao mesmo tempo (varios nucleos)
from scipy.stats import binomtest     # teste binomial (usado tambem para o McNemar)
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

# Nomes "bonitos" para as tabelas
NOME_MODELO = {"logistica": "Regressao Logistica", "random_forest": "Random Forest",
               "rede_neural": "Rede Neural (MLP)"}
BASES = {"sempre_alta": "Sempre alta", "majoritaria": "Classe majoritaria",
         "persistencia": "Persistencia"}
# Cada combinacao conjunto x modelo vira uma "chave", ex.: "tecnicos__logistica".
# (list comprehension: monta a lista percorrendo os dois loops)
CHAVES = [f"{c}__{m}" for c in config.CONJUNTOS for m in config.MODELOS]


# ===========================================================================
# OS MODELOS
# ===========================================================================
def novo_modelo(nome):
    """Cria um modelo NOVO (ainda nao treinado) com hiperparametros fixos."""
    if nome == "logistica":
        # Pipeline = "linha de montagem": primeiro padroniza as variaveis
        # (StandardScaler deixa cada uma com media 0 e desvio 1, para que indicadores
        # com escalas diferentes, como CCI ~100 e ROC ~0,01, pesem de forma comparavel)
        # e depois aplica a Regressao Logistica.
        # A Logistica calcula uma soma ponderada dos indicadores e a transforma numa
        # probabilidade entre 0 e 1 (funcao sigmoide).
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if nome == "random_forest":
        # Random Forest = 300 arvores de decisao, cada uma treinada com uma amostra
        # sorteada dos dados e um subconjunto sorteado de indicadores; no fim, votam.
        #   max_depth=5          cada arvore faz no maximo 5 perguntas em sequencia
        #   min_samples_leaf=50  cada "folha" (resposta final) precisa de >= 50 dias
        #   max_features="sqrt"  cada divisao sorteia ~raiz(10) = 3 indicadores
        # Arvores rasas e folhas grandes evitam "decorar" o ruido do mercado (overfitting).
        # Arvores nao precisam de padronizacao, por isso nao ha StandardScaler aqui.
        return RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=50,
                                      max_features="sqrt", n_jobs=1,
                                      random_state=config.SEMENTE)
    if nome == "rede_neural":
        # (extensao opcional) Rede neural com 1 camada oculta de 10 neuronios e
        # otimizador quasi-Newton L-BFGS.
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(10,), alpha=0.01, solver="lbfgs", max_iter=2000,
            random_state=config.SEMENTE))
    raise ValueError(nome)


# ===========================================================================
# VALIDACAO WALK-FORWARD
# ===========================================================================
def walk_forward(n):
    """Gera as janelas (fold, linhas_de_treino, linhas_de_teste) para uma serie de n dias.

    Exemplo com TREINO_INICIAL=504 e JANELA_TESTE=126:
      fold 0: treino = dias 0..503    teste = dias 504..629
      fold 1: treino = dias 0..629    teste = dias 630..755
      ...
    O treino sempre comeca no dia 0 e vai crescendo ("janela expansiva").

    "yield" transforma a funcao num gerador: ela entrega uma janela por vez,
    a cada volta do "for" que a usa.
    """
    h, inicio, fold = config.HORIZONTE, config.TREINO_INICIAL, 0
    while inicio < n:
        fim = min(inicio + config.JANELA_TESTE, n)
        # O alvo do dia i depende do preco do dia i+h. Para o treino nao conter nenhuma
        # informacao do periodo de teste, so entram dias com i+h <= inicio.
        # np.arange(a, b) = [a, a+1, ..., b-1]
        yield fold, np.arange(0, inicio - h + 1), np.arange(inicio, fim)
        inicio, fold = fim, fold + 1


def modelar_ativo(df, ultimo):
    """Roda o walk-forward completo para UM ativo.

    df     = todas as linhas desse ativo no dataset
    ultimo = a linha do ultimo dia (para a previsao "de hoje")
    Devolve: (previsoes nos dias de teste, importancias das variaveis, previsao atual)
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    df = df.sort_values("data").reset_index(drop=True)   # ordem cronologica, indice 0..n-1
    y = df["alvo"].astype(int).values                    # a resposta certa (0 ou 1)
    saidas, imps = [], []

    for fold, tr, te in walk_forward(len(df)):
        if len(np.unique(y[tr])) < 2:     # treino so com altas (ou so quedas): impossivel
            continue
        # tabela de saida: informacoes dos dias de teste
        out = df.loc[te, ["data", "ativo", "tipo", "periodo", "regime_vol", "alvo",
                          "ret_futuro", "cdi_futuro"]].copy()
        out["fold"] = fold

        for conj, variaveis in config.CONJUNTOS.items():
            X = df[variaveis].values      # matriz de indicadores (linhas = dias)
            for m in config.MODELOS:
                # .fit(...) TREINA o modelo SO com as linhas de treino
                mod = novo_modelo(m).fit(X[tr], y[tr])
                # .predict_proba(...) devolve [prob_baixa, prob_alta] para cada dia de teste;
                # [:, 1] pega a coluna da probabilidade de ALTA
                prob = mod.predict_proba(X[te])[:, 1]
                out[f"prob_{conj}__{m}"] = prob
                out[f"pred_{conj}__{m}"] = (prob >= 0.5).astype(int)   # >= 50% -> preve alta
                # guarda o que cada modelo "aprendeu" sobre os indicadores
                if m == "random_forest":
                    imps.append(("rf", conj, pd.Series(mod.feature_importances_, variaveis)))
                elif m == "logistica":
                    # mod[-1] = ultimo passo do pipeline (a regressao); coef_ = pesos
                    imps.append(("lr", conj, pd.Series(mod[-1].coef_[0], variaveis)))

        # linhas de base
        out["pred_sempre_alta"] = 1
        out["pred_majoritaria"] = int(y[tr].mean() >= 0.5)                 # moda do treino
        out["pred_persistencia"] = (df.loc[te, "roc"].values > 0).astype(int)  # hoje subiu?
        saidas.append(out)

    # previsao para o PROXIMO pregao: modelos treinados com TODO o historico conhecido
    atual = {}
    if not ultimo.empty:
        atual = {"ativo": df["ativo"].iloc[0], "data_referencia": ultimo["data"].iloc[0].date()}
        for conj, variaveis in config.CONJUNTOS.items():
            for m in config.MODELOS:
                mod = novo_modelo(m).fit(df[variaveis].values, y)
                atual[f"prob_alta_{conj}__{m}"] = mod.predict_proba(ultimo[variaveis].values)[0, 1]
    prev = pd.concat(saidas) if saidas else pd.DataFrame()
    return prev, imps, atual


# ===========================================================================
# METRICAS E TESTES ESTATISTICOS
# ===========================================================================
def rotulo(chave):
    """'tecnicos__logistica' -> ('tecnicos', 'Regressao Logistica')."""
    if chave in BASES:
        return "-", BASES[chave]
    conj, m = chave.split("__")
    return conj, NOME_MODELO[m]


def metricas(g):
    """Calcula as metricas de todos os modelos e linhas de base num grupo de previsoes g.

    acuracia             % de dias em que acertou a direcao
    acuracia_balanceada  media do acerto nas altas e nas quedas (justa se houver mais altas)
    precisao             quando previu alta, % de vezes que subiu mesmo
    recall               dos dias que subiram, % que o modelo previu como alta
    f1                   media harmonica de precisao e recall
    auc                  capacidade de ordenar dias de alta acima dos de queda pela
                         probabilidade (0,5 = acaso; 1,0 = perfeito)
    p_valor_vs_sempre_alta  teste binomial: chance de acertar tanto quanto acertou se,
                         na verdade, o modelo nao fosse melhor que "sempre alta"
    """
    y = g["alvo"].astype(int)
    acc_base = accuracy_score(y, g["pred_sempre_alta"])
    linhas = []
    for chave in CHAVES + list(BASES):
        p = g[f"pred_{chave}"]
        conj, modelo = rotulo(chave)
        lin = {"conjunto": conj, "modelo": modelo, "n": len(y),
               "acuracia": accuracy_score(y, p),
               "acuracia_balanceada": balanced_accuracy_score(y, p),
               "precisao": precision_score(y, p, zero_division=0),
               "recall": recall_score(y, p, zero_division=0),
               "f1": f1_score(y, p, zero_division=0),
               "auc": np.nan, "p_valor_vs_sempre_alta": np.nan}
        if chave in CHAVES:                 # AUC e p-valor so para os modelos de verdade
            if y.nunique() == 2:
                lin["auc"] = roc_auc_score(y, g[f"prob_{chave}"])
            # H0 (hipotese nula): a taxa de acerto do modelo e igual a de "sempre alta".
            # "greater" = teste unilateral: so interessa se o modelo for MELHOR.
            lin["p_valor_vs_sempre_alta"] = binomtest(int((p == y).sum()), len(y), acc_base,
                                                      alternative="greater").pvalue
        linhas.append(lin)
    return pd.DataFrame(linhas)


def mcnemar(g, a, b):
    """Teste exato de McNemar entre os modelos a e b (Dietterich, 1998).

    Ideia: dias em que os DOIS acertam ou os DOIS erram nao dizem nada sobre quem e
    melhor. So importam os dias em que UM acertou e o OUTRO errou. Se os modelos fossem
    equivalentes, esses dias se dividiriam meio a meio (como cara ou coroa). O p-valor
    mede a chance de uma divisao tao desigual quanto a observada acontecer por acaso.
    """
    y = g["alvo"].astype(int)
    acerta_a, acerta_b = g[f"pred_{a}"] == y, g[f"pred_{b}"] == y
    so_a = int((acerta_a & ~acerta_b).sum())     # dias em que so o modelo a acertou
    so_b = int((~acerta_a & acerta_b).sum())     # dias em que so o modelo b acertou
    p = binomtest(so_a, so_a + so_b, 0.5).pvalue if so_a + so_b > 0 else 1.0
    return {"modelo_a": rotulo(a)[1], "modelo_b": rotulo(b)[1],
            "acuracia_a": acerta_a.mean(), "acuracia_b": acerta_b.mean(),
            "so_a_acerta": so_a, "so_b_acerta": so_b, "p_valor_mcnemar": p}


def backtest(g):
    """Simulacao financeira: fica no ativo se o modelo preve alta; senao, fica no CDI.

    pos = 1 (comprado no ativo) ou 0 (no caixa, rendendo CDI).
    retorno do dia = pos x retorno do ativo + (1 - pos) x CDI - custo das trocas.
    """
    g = g.sort_values("data").iloc[::config.HORIZONTE]   # periodos sem sobreposicao
    anual = 252 / config.HORIZONTE                       # periodos por ano
    curvas, linhas = {"data": g["data"].values}, []
    # Estrategias comparadas: duas de referencia + uma por modelo
    estrategias = [("Comprar e manter", None), ("CDI", "cdi")] + \
                  [(f"{rotulo(k)[1]} [{rotulo(k)[0]}]", k) for k in CHAVES]
    for nome, chave in estrategias:
        if chave is None:
            pos = pd.Series(1, index=g.index)      # sempre no ativo
        elif chave == "cdi":
            pos = pd.Series(0, index=g.index)      # sempre no caixa
        else:
            pos = g[f"pred_{chave}"]               # segue o modelo
        # numero de trocas: 1 sempre que a posicao muda de um dia para o outro
        trocas = pos.diff().abs().fillna(pos.iloc[0]) if chave not in (None, "cdi") else 0
        ret = pos * g["ret_futuro"] + (1 - pos) * g["cdi_futuro"] - trocas * config.CUSTO_POR_TROCA
        patrimonio = (1 + ret).cumprod()           # R$ 1 investido no inicio vira quanto?
        excesso, anos = ret - g["cdi_futuro"], len(ret) / anual
        # mesma estrategia sem custo de troca: mostra o quanto os custos pesam no resultado
        ret_sem_custo = pos * g["ret_futuro"] + (1 - pos) * g["cdi_futuro"]
        linhas.append({
            "estrategia": nome,
            "retorno_total": patrimonio.iloc[-1] - 1,
            "retorno_anual": patrimonio.iloc[-1] ** (1 / anos) - 1,
            "retorno_anual_sem_custo": (1 + ret_sem_custo).prod() ** (1 / anos) - 1,
            "volatilidade_anual": ret.std() * np.sqrt(anual),
            # Sharpe: retorno acima do CDI por unidade de risco (quanto maior, melhor)
            "sharpe": excesso.mean() / ret.std() * np.sqrt(anual) if ret.std() > 0 else np.nan,
            # drawdown maximo: a pior queda desde um pico (o "susto" maximo)
            "drawdown_maximo": (patrimonio / patrimonio.cummax() - 1).min(),
            "tempo_no_ativo": pos.mean(),
            "trocas": int(pos.diff().abs().sum()),
        })
        curvas[nome] = patrimonio.values
    return pd.DataFrame(linhas), pd.DataFrame(curvas)


def retorno_anual(pos, g, custo):
    """Retorno anualizado de ficar no ativo quando pos = 1 e no CDI quando pos = 0."""
    pos = np.asarray(pos, dtype=float)
    trocas = np.abs(np.diff(pos, prepend=0.0))      # a entrada inicial tambem conta
    ret = pos * g["ret_futuro"].values + (1 - pos) * g["cdi_futuro"].values - trocas * custo
    anos = len(ret) / (252 / config.HORIZONTE)
    return np.prod(1 + ret) ** (1 / anos) - 1


def sensibilidade_custos(g):
    """Refaz o backtest de cada modelo com cada custo de config.CUSTOS_SENSIBILIDADE."""
    g = g.sort_values("data").iloc[::config.HORIZONTE]
    manter = retorno_anual(np.ones(len(g)), g, 0.0)
    cdi = retorno_anual(np.zeros(len(g)), g, 0.0)
    linhas = []
    for k in CHAVES:
        for custo in config.CUSTOS_SENSIBILIDADE:
            r = retorno_anual(g[f"pred_{k}"].values, g, custo)
            linhas.append({"conjunto": rotulo(k)[0], "modelo": rotulo(k)[1], "custo": custo,
                           "retorno_anual": r, "retorno_comprar_e_manter": manter,
                           "retorno_cdi": cdi, "supera_comprar_e_manter": r > manter,
                           "supera_cdi": r > cdi})
    return pd.DataFrame(linhas)


def estrategias_aleatorias(g, semente):
    """Compara o modelo (sem custos) com estrategias aleatorias de mesma exposicao.

    percentil = % das estrategias aleatorias que o modelo superou. Perto de 50% significa
    que o modelo se comportou como uma escolha ao acaso com o mesmo tempo no ativo.
    """
    g = g.sort_values("data").iloc[::config.HORIZONTE]
    rng = np.random.default_rng(semente)
    manter = retorno_anual(np.ones(len(g)), g, 0.0)
    linhas = []
    for k in CHAVES:
        pos = g[f"pred_{k}"].values
        exposicao = pos.mean()
        modelo = retorno_anual(pos, g, 0.0)
        sorteios = rng.random((config.N_ESTRATEGIAS_ALEATORIAS, len(g))) < exposicao
        aleatorias = np.array([retorno_anual(s, g, 0.0) for s in sorteios])
        linhas.append({"conjunto": rotulo(k)[0], "modelo": rotulo(k)[1],
                       "exposicao_ao_ativo": exposicao, "retorno_anual_sem_custo": modelo,
                       "mediana_aleatorias": np.median(aleatorias),
                       "percentil_entre_aleatorias": (aleatorias < modelo).mean(),
                       "aleatorias_que_superam_comprar_e_manter": (aleatorias > manter).mean()})
    return pd.DataFrame(linhas)


# ===========================================================================
# PROGRAMA PRINCIPAL
# ===========================================================================
def main():
    inicio = time.time()
    config.RESULTADOS.mkdir(parents=True, exist_ok=True)
    dados = pd.read_parquet(config.DADOS_PROCESSADOS / "dataset.parquet")
    ultimo = pd.read_parquet(config.DADOS_PROCESSADOS / "ultimo_dia.parquet")
    ativos = sorted(dados["ativo"].unique())
    print(f"Modelando {len(ativos)} ativos x {len(CHAVES)} combinacoes modelo/conjunto "
          f"(em paralelo; pode levar alguns minutos)...")

    # ---- 1. WALK-FORWARD DE TODOS OS ATIVOS EM PARALELO ----
    # Parallel(n_jobs=-1) usa todos os nucleos do processador; delayed(f)(args) agenda f(args).
    # Diferente do download (threads, que esperam a rede), aqui o trabalho e CALCULO,
    # entao usamos processos em nucleos diferentes.
    res = Parallel(n_jobs=-1, verbose=5)(
        delayed(modelar_ativo)(dados[dados["ativo"] == a], ultimo[ultimo["ativo"] == a])
        for a in ativos)
    # res = lista com um resultado (previsoes, importancias, atual) por ativo

    prev = pd.concat([r[0] for r in res if not r[0].empty], ignore_index=True)
    prev.to_parquet(config.RESULTADOS / "previsoes.parquet", index=False)
    pd.DataFrame([r[2] for r in res if r[2]]).to_csv(
        config.RESULTADOS / "previsao_atual.csv", index=False)

    # ---- 2. IMPORTANCIA DAS VARIAVEIS (media sobre todos os folds e ativos) ----
    imp = {}
    for tipo, conj, s in (x for r in res for x in r[1]):
        imp.setdefault((tipo, conj), []).append(s)   # agrupa por (modelo, conjunto)
    tab_imp = []
    for conj in config.CONJUNTOS:
        rf = pd.concat(imp[("rf", conj)], axis=1).mean(axis=1)
        lr = pd.concat(imp[("lr", conj)], axis=1).mean(axis=1)
        tab_imp.append(pd.DataFrame({"conjunto": conj, "variavel": rf.index,
                                     "importancia_rf": rf.values, "coef_logistica": lr.values}))
    pd.concat(tab_imp).sort_values(["conjunto", "importancia_rf"], ascending=[True, False]) \
        .to_csv(config.RESULTADOS / "importancia_variaveis.csv", index=False)

    # ---- 3. METRICAS POR SEGMENTO ----
    # groupby("coluna") separa as previsoes em grupos (ex.: validacao / reservado)
    segmentos = [("geral", "todos", prev)]
    segmentos += [("periodo", k, g) for k, g in prev.groupby("periodo")]
    segmentos += [("tipo", k, g) for k, g in prev.groupby("tipo")]
    segmentos += [("regime_vol", k, g) for k, g in prev.groupby("regime_vol")]
    resumo = pd.concat([metricas(g).assign(segmento=s, valor=v) for s, v, g in segmentos])
    resumo.to_csv(config.RESULTADOS / "metricas_resumo.csv", index=False)

    pd.concat([metricas(g).assign(ativo=a, tipo=g["tipo"].iloc[0])
               for a, g in prev.groupby("ativo")]) \
        .to_csv(config.RESULTADOS / "metricas_por_ativo.csv", index=False)

    prev["ano"] = pd.to_datetime(prev["data"]).dt.year
    pd.concat([metricas(g).assign(ano=a) for a, g in prev.groupby("ano")]) \
        .to_csv(config.RESULTADOS / "metricas_por_ano.csv", index=False)

    # ---- 4. COMPARACAO ENTRE MODELOS (McNemar): A PERGUNTA CENTRAL ----
    comp = []
    for conj in config.CONJUNTOS:
        chaves = [f"{conj}__{m}" for m in config.MODELOS]
        for a, b in combinations(chaves, 2):        # todos os pares (hoje: LR x RF)
            # "*" desempacota as listas dentro da lista principal
            for seg, val, g in [("geral", "todos", prev),
                                *[("periodo", k, x) for k, x in prev.groupby("periodo")],
                                *[("ano", str(k), x) for k, x in prev.groupby("ano")],
                                *[("regime_vol", k, x) for k, x in prev.groupby("regime_vol")]]:
                comp.append({"conjunto": conj, "segmento": seg, "valor": val, **mcnemar(g, a, b)})
    # o conjunto ampliado melhora cada modelo? (so se os dois conjuntos estiverem ativos)
    for m in (config.MODELOS if {"tecnicos", "ampliado"} <= set(config.CONJUNTOS) else []):
        a, b = f"ampliado__{m}", f"tecnicos__{m}"
        comp.append({"conjunto": "ampliado vs tecnicos", "segmento": "geral", "valor": "todos",
                     **mcnemar(prev, a, b)})
    comp = pd.DataFrame(comp)
    comp.to_csv(config.RESULTADOS / "comparacao_modelos.csv", index=False)

    # ---- 5. ESTABILIDADE: em quantos anos/ativos/regimes a RF superou a Logistica ----
    estab = []
    for conj in (config.CONJUNTOS if {"logistica", "random_forest"} <= set(config.MODELOS) else []):
        rf, lr = f"pred_{conj}__random_forest", f"pred_{conj}__logistica"
        for nivel in ("ano", "ativo", "regime_vol"):
            acc = prev.groupby(nivel).apply(
                lambda g: pd.Series({"rf": (g[rf] == g["alvo"]).mean(),
                                     "lr": (g[lr] == g["alvo"]).mean()}), include_groups=False)
            estab.append({"conjunto": conj, "nivel": nivel, "grupos": len(acc),
                          "rf_melhor": int((acc["rf"] > acc["lr"]).sum()),
                          "diferenca_media_pp": round((acc["rf"] - acc["lr"]).mean() * 100, 2)})
    pd.DataFrame(estab, columns=["conjunto", "nivel", "grupos", "rf_melhor",
                                 "diferenca_media_pp"]).to_csv(config.RESULTADOS / "estabilidade.csv", index=False)

    # ---- 6. MATRIZES DE CONFUSAO (quais erros sao mais frequentes) ----
    #                    previu baixa        previu alta
    # caiu de verdade    verdadeiro_baixa    falso_alta
    # subiu de verdade   falsa_baixa         verdadeira_alta
    mc = []
    y = prev["alvo"].astype(int)
    for chave in CHAVES:
        (vn, fp), (fn, vp) = confusion_matrix(y, prev[f"pred_{chave}"], labels=[0, 1])
        conj, modelo = rotulo(chave)
        mc.append({"conjunto": conj, "modelo": modelo, "verdadeiro_baixa": vn,
                   "falso_alta": fp, "falsa_baixa": fn, "verdadeira_alta": vp})
    pd.DataFrame(mc).to_csv(config.RESULTADOS / "matrizes_confusao.csv", index=False)

    # ---- 7. BACKTEST (todo o periodo de teste e so o periodo reservado) ----
    bts, curvas = [], []
    for a, g in prev.groupby("ativo"):
        for per, gg in [("todo o teste", g), ("reservado", g[g["periodo"] == "reservado"])]:
            if len(gg) < 20:
                continue
            b, c = backtest(gg)
            bts.append(b.assign(ativo=a, tipo=g["tipo"].iloc[0], periodo=per))
            if per == "todo o teste":
                curvas.append(c.assign(ativo=a))
    pd.concat(bts).to_csv(config.RESULTADOS / "backtest_por_ativo.csv", index=False)
    pd.concat(curvas).to_parquet(config.RESULTADOS / "curvas_backtest.parquet", index=False)

    # ---- 7b. SENSIBILIDADE AOS CUSTOS E ESTRATEGIAS ALEATORIAS (todo o teste) ----
    sens, alea = [], []
    for i, (a, g) in enumerate(prev.groupby("ativo")):   # ordem alfabetica: semente estavel
        sens.append(sensibilidade_custos(g).assign(ativo=a, tipo=g["tipo"].iloc[0]))
        alea.append(estrategias_aleatorias(g, config.SEMENTE + i).assign(ativo=a, tipo=g["tipo"].iloc[0]))
    sens = pd.concat(sens)
    sens.to_csv(config.RESULTADOS / "backtest_sensibilidade.csv", index=False)
    pd.concat(alea).to_csv(config.RESULTADOS / "backtest_aleatorio.csv", index=False)

    # ---- 8. RESUMO NA TELA ----
    cols = ["segmento", "valor", "conjunto", "modelo", "n", "acuracia", "auc",
            "p_valor_vs_sempre_alta"]
    print("\nResumo:")
    print(resumo[resumo["segmento"].isin(["geral", "periodo"])][cols].round(3).to_string(index=False))
    print("\nBacktest: em quantos ativos o modelo supera comprar e manter, por custo:")
    print(sens.groupby(["modelo", "custo"])["supera_comprar_e_manter"].sum()
          .unstack("custo").to_string())
    print("\nRandom Forest x Logistica (McNemar):")
    print(comp[(comp["modelo_a"] == "Regressao Logistica") & (comp["modelo_b"] == "Random Forest")
               & comp["segmento"].isin(["geral", "periodo"])]
          [["conjunto", "segmento", "valor", "acuracia_a", "acuracia_b", "p_valor_mcnemar"]]
          .round(4).to_string(index=False))
    print(f"\nConcluido em {(time.time() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
