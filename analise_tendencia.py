"""
analise_tendencia.py - TENDENCIA LINEAR x EXPONENCIAL E R2 DOS CLASSIFICADORES
==============================================================================

Analise complementar, fora do experimento principal. Responde a duas perguntas:

1. A tendencia de preco de cada ativo e mais bem descrita por uma reta ou por uma
   exponencial?
     linear:      P = a + b*t
     exponencial: P = A*exp(B*t)  ->  ln(P) = ln(A) + B*t  (reta no logaritmo)
   t = tempo em anos desde o primeiro pregao. O R2 das duas e calculado NA MESMA
   ESCALA (a do preco), para que sejam comparaveis; o R2 da exponencial na escala
   do logaritmo tambem e informado, porque e o que a linha de tendencia do Excel mostra.
   A analise e repetida com precos diarios, mensais (ultimo preco do mes) e anuais
   (ultimo preco do ano).

2. Quanto do resultado (alta/baixa) os classificadores explicam fora da amostra?
   O R2 comum nao se aplica a um alvo 0/1. Usam-se o pseudo-R2 de McFadden e o
   Brier skill, contra um modelo nulo que preve sempre a proporcao de altas do treino
   de cada janela.

Inclui um controle: o R2 de uma reta ajustada a passeios aleatorios simulados (sem
tendencia e sem previsibilidade), para mostrar que R2 alto de preco x tempo nao indica
previsibilidade (regressao espuria).

Uso:   python analise_tendencia.py      (rode depois do atualizar.py)
Saida: resultados/tendencia_r2.csv, resultados/tendencia_passeio_aleatorio.csv,
       resultados/r2_classificadores.csv e resultados/teste_r2_linear_exponencial.xlsx
"""
import importlib

import numpy as np
import pandas as pd

import config

modelar = importlib.import_module("03_modelar")

N_SIMULACOES = 2000          # passeios aleatorios simulados
FREQUENCIAS = {"diaria": None, "mensal": "ME", "anual": "YE"}


def r2(y, y_ajustado):
    """Coeficiente de determinacao: 1 - SQ residual / SQ total."""
    return 1 - np.sum((y - y_ajustado) ** 2) / np.sum((y - np.mean(y)) ** 2)


def ajustar(t, preco):
    """Ajusta a reta e a exponencial por minimos quadrados e devolve os R2."""
    b, a = np.polyfit(t, preco, 1)                 # P = a + b*t
    B, lnA = np.polyfit(t, np.log(preco), 1)       # ln P = ln A + B*t
    linear = a + b * t
    exponencial = np.exp(lnA + B * t)
    return {
        "a_linear": a, "b_linear": b,
        "A_exp": np.exp(lnA), "B_exp": B,
        "r2_linear": r2(preco, linear),
        "r2_exp_escala_preco": r2(preco, exponencial),
        "r2_exp_escala_log": r2(np.log(preco), lnA + B * t),
        "crescimento_anual_exp": np.exp(B) - 1,
    }


def serie(g, freq):
    """Precos do ativo na frequencia pedida (ultimo preco de cada periodo)."""
    s = g.set_index("data")["preco"].sort_index()
    if FREQUENCIAS[freq]:
        s = s.resample(FREQUENCIAS[freq]).last().dropna()
    t = (s.index - s.index[0]).days.values / 365.25
    return s, t


def tendencias(dados):
    linhas = []
    for ativo, g in dados.groupby("ativo"):
        for freq in FREQUENCIAS:
            s, t = serie(g, freq)
            linhas.append({"ativo": ativo, "tipo": g["tipo"].iloc[0], "frequencia": freq,
                           "n": len(s), **ajustar(t, s.values)})
    return pd.DataFrame(linhas)


def passeio_aleatorio(dados, ativo="BOVA11"):
    """R2 da reta em passeios aleatorios com a volatilidade diaria do ativo."""
    g = dados[dados["ativo"] == ativo].sort_values("data")
    sd = np.diff(np.log(g["preco"].values)).std()
    n = len(g)
    t = (g["data"] - g["data"].iloc[0]).dt.days.values / 365.25
    rng = np.random.default_rng(config.SEMENTE)
    r2s = []
    for _ in range(N_SIMULACOES):
        preco = 100 * np.exp(np.cumsum(rng.normal(0, sd, n)))   # retornos de media zero
        r2s.append(ajustar(t, preco)["r2_linear"])
    r2s = np.array(r2s)
    exemplo = 100 * np.exp(np.cumsum(np.random.default_rng(config.SEMENTE).normal(0, sd, n)))
    resumo = pd.DataFrame([{
        "ativo_referencia": ativo, "volatilidade_diaria": sd, "pregoes": n,
        "simulacoes": N_SIMULACOES, "r2_mediano": np.median(r2s),
        "proporcao_r2_acima_0_5": np.mean(r2s > 0.5),
    }])
    return resumo, g["data"].values, exemplo


def classificadores(dados, previsoes):
    """Pseudo-R2 de McFadden e Brier skill fora da amostra, contra o modelo nulo."""
    base = {}
    for ativo, g in dados.sort_values("data").groupby("ativo"):
        y = g["alvo"].values
        for fold, tr, _ in modelar.walk_forward(len(g)):
            base[(ativo, fold)] = y[tr].mean()       # proporcao de altas no treino
    p = previsoes.copy()
    p["prob_nulo"] = [base[(a, f)] for a, f in zip(p["ativo"], p["fold"])]
    y = p["alvo"].astype(float).values

    def log_ver(q):
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    def brier(q):
        return np.mean((q - y) ** 2)

    colunas = {"Regressao Logistica": "prob_tecnicos__logistica",
               "Random Forest": "prob_tecnicos__random_forest",
               "Nulo (proporcao de altas do treino)": "prob_nulo"}
    ll0, bs0 = log_ver(p["prob_nulo"].values), brier(p["prob_nulo"].values)
    linhas = []
    for nome, col in colunas.items():
        q = p[col].values
        linhas.append({"modelo": nome, "n": len(y),
                       "acuracia": np.mean((q >= 0.5) == (y == 1)),
                       "brier": brier(q), "brier_skill": 1 - brier(q) / bs0,
                       "log_verossimilhanca": log_ver(q),
                       "pseudo_r2_mcfadden": 1 - log_ver(q) / ll0})
    return pd.DataFrame(linhas), p


def main():
    dados = pd.read_parquet(config.DADOS_PROCESSADOS / "dataset.parquet")
    previsoes = pd.read_parquet(config.RESULTADOS / "previsoes.parquet")

    tend = tendencias(dados)
    pa, datas_pa, exemplo_pa = passeio_aleatorio(dados)
    clas, prev = classificadores(dados, previsoes)

    tend.to_csv(config.RESULTADOS / "tendencia_r2.csv", index=False)
    pa.to_csv(config.RESULTADOS / "tendencia_passeio_aleatorio.csv", index=False)
    clas.to_csv(config.RESULTADOS / "r2_classificadores.csv", index=False)

    import planilha_tendencia
    planilha_tendencia.exportar(dados, tend, pa, datas_pa, exemplo_pa, clas, prev,
                                config.RESULTADOS / "teste_r2_linear_exponencial.xlsx")

    pd.set_option("display.width", 160)
    d = tend[tend["frequencia"] == "diaria"]
    print("R2 da tendencia (precos diarios, escala do preco):")
    print(d[["ativo", "r2_linear", "r2_exp_escala_preco", "r2_exp_escala_log",
             "crescimento_anual_exp"]].round(3).to_string(index=False))
    print("\nMedia por frequencia:")
    print(tend.groupby("frequencia")[["r2_linear", "r2_exp_escala_preco"]].mean().round(3))
    print("\nPasseio aleatorio:", pa.round(3).to_dict("records")[0])
    print("\nClassificadores (fora da amostra):")
    print(clas.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
