"""
Testes automatizados do pipeline.  Rodar na pasta do projeto:  python -m pytest -v

Cada teste verifica uma propriedade que o metodo exige. Se algum falhar, os
resultados do TCC nao seriam confiaveis.
"""
import numpy as np
import pandas as pd
import pytest

import config


# ---------------------------------------------------------------------------
# Dados sinteticos: uma serie de precos aleatoria, com OHLC coerente
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def precos():
    rng = np.random.default_rng(0)
    datas = pd.bdate_range("2017-01-02", periods=900)
    fechamento = 50 * np.exp(np.cumsum(rng.normal(0, 0.015, len(datas))))
    abertura = fechamento * (1 + rng.normal(0, 0.003, len(datas)))
    maxima = np.maximum(fechamento, abertura) * (1 + np.abs(rng.normal(0, 0.005, len(datas))))
    minima = np.minimum(fechamento, abertura) * (1 - np.abs(rng.normal(0, 0.005, len(datas))))
    return pd.DataFrame({"abertura": abertura, "maxima": maxima, "minima": minima,
                         "fechamento": fechamento, "volume": 1e6},
                        index=pd.Index(datas, name="data"))


@pytest.fixture(scope="module")
def vix(precos):
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(0, 0.05, len(precos)), index=precos.index)


# ---------------------------------------------------------------------------
# 1. Nenhum indicador pode usar informacao futura (verificacao de vazamento temporal)
# ---------------------------------------------------------------------------
def test_indicadores_nao_usam_o_futuro(prep, precos, vix):
    completo = prep.calcular_variaveis(precos, vix)
    for dia in precos.index[300::50]:
        # recalcula como se a serie terminasse neste dia (o futuro nao existe)
        cortado = prep.calcular_variaveis(precos.loc[:dia], vix.loc[:dia])
        a = completo.loc[dia, config.TODAS_VARIAVEIS].astype(float).values
        b = cortado.loc[dia, config.TODAS_VARIAVEIS].astype(float).values
        assert np.allclose(a, b, equal_nan=True), f"indicador diferente em {dia.date()}"


def test_vix_usa_somente_o_dia_anterior(prep):
    datas_b3 = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"])
    var_vix = pd.Series([0.10, 0.20, 0.30], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                        name="vix_var")
    alinhado = prep.alinhar_vix(var_vix, datas_b3)
    # em 04/01 o modelo so pode conhecer o VIX de 03/01 (0.20), nunca o do proprio dia
    assert list(alinhado.values) == [0.10, 0.20, 0.30]


# ---------------------------------------------------------------------------
# 2. Alvo: 1 quando o preco do dia seguinte sobe; nunca usado como variavel
# ---------------------------------------------------------------------------
def test_alvo_fora_das_variaveis():
    proibidas = {"alvo", "ret_futuro", "cdi_futuro"}
    for nome, variaveis in config.CONJUNTOS.items():
        assert not proibidas & set(variaveis), f"variavel do futuro no conjunto {nome}"


def test_definicao_do_alvo(precos):
    c = precos["fechamento"]
    ret_futuro = c.shift(-1) / c - 1
    alvo = (ret_futuro > 0).astype(int)
    assert alvo.iloc[0] == int(c.iloc[1] > c.iloc[0])


# ---------------------------------------------------------------------------
# 3. Walk-forward: treino antes do teste, sem sobreposicao, do fim do treino inicial ao ultimo dia
# ---------------------------------------------------------------------------
def test_walk_forward(modelar):
    n = 2000
    janelas = list(modelar.walk_forward(n))
    testados = np.concatenate([te for _, _, te in janelas])
    assert testados[0] == config.TREINO_INICIAL           # primeiro teste apos o treino inicial
    assert testados[-1] == n - 1                          # vai ate o ultimo dia
    assert len(testados) == len(set(testados))            # nenhum dia testado duas vezes
    for _, tr, te in janelas:
        # o ultimo dia de treino + horizonte nao pode alcancar o primeiro dia de teste
        assert tr.max() + config.HORIZONTE <= te.min()
        assert tr.min() == 0                              # janela expansiva


# ---------------------------------------------------------------------------
# 4. Teste de McNemar com um exemplo calculado a mao
# ---------------------------------------------------------------------------
def test_mcnemar(modelar):
    a, b = "tecnicos__logistica", "tecnicos__random_forest"
    g = pd.DataFrame({"alvo":      [1, 1, 1, 1, 0, 0, 0, 0],
                      f"pred_{a}": [1, 1, 1, 1, 0, 0, 0, 0],   # acerta tudo
                      f"pred_{b}": [0, 0, 1, 1, 1, 0, 0, 0]})  # erra 3
    r = modelar.mcnemar(g, a, b)
    assert r["so_a_acerta"] == 3 and r["so_b_acerta"] == 0
    assert r["p_valor_mcnemar"] == pytest.approx(0.25)    # 2 x (1/2)^3


# ---------------------------------------------------------------------------
# 5. Reprodutibilidade: o mesmo treino gera exatamente as mesmas previsoes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nome", config.MODELOS)
def test_modelos_reprodutiveis(modelar, nome):
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(600, len(config.VARIAVEIS_TECNICAS))), rng.integers(0, 2, 600)
    p1 = modelar.novo_modelo(nome).fit(X, y).predict_proba(X)[:, 1]
    p2 = modelar.novo_modelo(nome).fit(X, y).predict_proba(X)[:, 1]
    assert np.array_equal(p1, p2)


# ---------------------------------------------------------------------------
# Backtest: a analise de sensibilidade usa a mesma conta do backtest principal
# ---------------------------------------------------------------------------
def test_sensibilidade_igual_ao_backtest(modelar):
    rng = np.random.default_rng(3)
    n = 300
    g = pd.DataFrame({"data": pd.bdate_range("2020-01-01", periods=n),
                      "ret_futuro": rng.normal(0, 0.01, n),
                      "cdi_futuro": np.full(n, 0.0004)})
    for k in modelar.CHAVES:
        g[f"pred_{k}"] = rng.integers(0, 2, n)
    principal, _ = modelar.backtest(g)
    sens = modelar.sensibilidade_custos(g)
    for k in modelar.CHAVES:
        conj, nome = modelar.rotulo(k)
        esperado = principal.loc[principal["estrategia"] == f"{nome} [{conj}]", "retorno_anual"].iloc[0]
        obtido = sens.loc[(sens["modelo"] == nome) & np.isclose(sens["custo"], config.CUSTO_POR_TROCA),
                          "retorno_anual"].iloc[0]
        assert np.isclose(esperado, obtido)


# ---------------------------------------------------------------------------
# Bootstrap em blocos: a diferenca observada e a diferenca real de acuracia
# ---------------------------------------------------------------------------
def test_bootstrap_blocos(modelar):
    rng = np.random.default_rng(5)
    datas = pd.bdate_range("2020-01-01", periods=200)
    g = pd.DataFrame({"data": np.repeat(datas, 3), "alvo": rng.integers(0, 2, 600)})
    g["pred_x"] = rng.integers(0, 2, 600)
    g["pred_y"] = 1
    r = modelar.bootstrap_blocos(g, "x", "y", semente=1)
    esperado = (g["pred_x"] == g["alvo"]).mean() - (g["pred_y"] == g["alvo"]).mean()
    assert np.isclose(r["diferenca"], esperado)
    assert r["ic95_inf"] <= r["diferenca"] <= r["ic95_sup"]
    assert 0 <= r["p_bilateral"] <= 1 and r["pregoes"] == 200
    # mesma semente -> mesmo resultado (reprodutibilidade)
    assert r == modelar.bootstrap_blocos(g, "x", "y", semente=1)
