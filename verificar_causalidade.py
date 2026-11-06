"""
Verificacao de causalidade com DADOS REAIS (complementa o teste automatizado).

O teste test_indicadores_nao_usam_o_futuro usa uma serie sintetica. Este script faz a
mesma verificacao com os precos reais ja baixados (dados/brutos): para tres ativos de
classes diferentes e quinze datas espalhadas pelo periodo, calcula as variaveis com a
serie COMPLETA e de novo com a serie CORTADA na data (como se o futuro nao existisse).
Se nenhuma variavel usa o futuro, os dois valores tem de ser identicos.

Como controle, o script cria de proposito uma variavel que olha o futuro (o retorno do
dia seguinte) e mostra que a verificacao a detecta, o que indica que o procedimento e
capaz de identificar o uso de informacao futura.

Uso (na pasta do projeto, com o .venv ativo):
    python verificar_causalidade.py

Saida: tabela na tela e o arquivo resultados/verificacao_causalidade.csv
"""
import importlib
import warnings

import numpy as np
import pandas as pd

import config

warnings.simplefilter("ignore")
prep = importlib.import_module("02_preparar_variaveis")   # nome comeca com numero

ATIVOS = ["BOVA11", "PETR4", "HGLG11"]   # um ETF, uma acao e um FII
N_DATAS = 15


def variaveis(px, var_vix, controle=False):
    """Calcula as variaveis do modelo (e, se pedido, a variavel de controle que vaza)."""
    df = prep.calcular_variaveis(px, prep.alinhar_vix(var_vix, px.index))
    if controle:
        # VARIAVEL PROPOSITALMENTE ERRADA: retorno de amanha (informacao futura)
        df["controle_futuro"] = px["fechamento"].pct_change().shift(-1)
    return df


def main():
    var_vix = prep.variacao_vix()
    ini, fim = pd.Timestamp(config.DATA_INICIO), pd.Timestamp(config.DATA_FIM)
    linhas = []
    for ativo in ATIVOS:
        px = prep.ler_precos(ativo)
        if px is None:
            print(f"{ativo}: sem dados em dados/brutos. Rode antes: python 01_baixar_dados.py (so baixa os brutos; nao altera a base congelada)")
            continue
        px = prep.tratar(px)
        completo = variaveis(px, var_vix, controle=True)
        # quinze datas espalhadas pelo periodo analisado (sem a ultima, que nao tem "amanha")
        datas_validas = completo.loc[ini:fim].dropna(subset=config.TODAS_VARIAVEIS).index[:-1]
        escolhidas = datas_validas[np.linspace(0, len(datas_validas) - 1, N_DATAS).round().astype(int)]
        for dia in escolhidas:
            vix_ate = var_vix.loc[:dia] if var_vix is not None else None
            cortado = variaveis(px.loc[:dia], vix_ate, controle=True)
            a = completo.loc[dia, config.TODAS_VARIAVEIS].astype(float).values
            b = cortado.loc[dia, config.TODAS_VARIAVEIS].astype(float).values
            ctrl_a = completo.loc[dia, "controle_futuro"]
            ctrl_b = cortado.loc[dia, "controle_futuro"]
            linhas.append({
                "ativo": ativo,
                "data": dia.date(),
                "variaveis_verificadas": len(config.TODAS_VARIAVEIS),
                "maior_diferenca": float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else 0.0,
                "variaveis_identicas": bool(np.allclose(a, b, equal_nan=True)),
                # o controle TEM de falhar: com a serie cortada, o "amanha" nao existe (NaN)
                "controle_detectado": not (np.isclose(ctrl_a, ctrl_b) if pd.notna(ctrl_b) else False),
            })
    res = pd.DataFrame(linhas)
    if res.empty:
        return
    pd.set_option("display.width", 120)
    print(res.to_string(index=False))
    saida = config.RAIZ / "resultados" / "verificacao_causalidade.csv"
    saida.parent.mkdir(exist_ok=True)
    res.to_csv(saida, index=False)

    ok = res["variaveis_identicas"].all()
    ctrl = res["controle_detectado"].all()
    print(f"\nDatas verificadas: {len(res)} ({res['ativo'].nunique()} ativos x {N_DATAS} datas)")
    print(f"Variaveis do modelo identicas com a serie cortada: {'SIM' if ok else 'NAO'} "
          f"(maior diferenca: {res['maior_diferenca'].max():.2e})")
    print(f"Variavel de controle (que olha o futuro) detectada em todas as datas: {'SIM' if ctrl else 'NAO'}")
    print(f"Arquivo salvo em: {saida}")


if __name__ == "__main__":
    main()
