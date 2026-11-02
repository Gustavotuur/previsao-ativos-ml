"""
Verificacao do calendario da base congelada: procura pregoes ausentes em cada ativo.

O alvo compara cada registro com o registro SEGUINTE do mesmo ativo. Se faltar um pregao
na serie de um ativo, o "dia seguinte" passa a ser dois (ou mais) pregoes depois. Este
script usa como calendario de referencia a uniao das datas de todos os ativos da base e
lista, para cada ativo, os pregoes que faltam e quantas previsoes do periodo de teste
foram afetadas.

Uso (na pasta do projeto, com o .venv ativo):
    python verificar_calendario.py

Saida: tabela na tela e o arquivo resultados/verificacao_calendario.csv
"""
import pandas as pd

import config

base = pd.read_parquet(config.DADOS_PROCESSADOS / "dataset.parquet", columns=["data", "ativo"])
base["data"] = pd.to_datetime(base["data"])
prev = pd.read_parquet(config.RAIZ / "resultados" / "previsoes.parquet", columns=["data", "ativo"])
prev["data"] = pd.to_datetime(prev["data"])

calendario = sorted(base["data"].unique())          # todos os pregoes que aparecem na base
posicao = {d: i for i, d in enumerate(calendario)}

linhas = []
for ativo, g in base.groupby("ativo"):
    datas = sorted(g["data"])
    teste = set(prev.loc[prev["ativo"] == ativo, "data"])
    for antes, depois in zip(datas[:-1], datas[1:]):
        salto = posicao[depois] - posicao[antes]
        if salto > 1:                                 # faltou pelo menos um pregao no meio
            faltam = [str(calendario[k].date()) for k in range(posicao[antes] + 1, posicao[depois])]
            linhas.append({"ativo": ativo, "ultimo_registro": antes.date(), "proximo_registro": depois.date(),
                           "pregoes_ausentes": ", ".join(faltam), "qtd_ausentes": len(faltam),
                           "previsao_afetada_no_teste": antes in teste})

res = pd.DataFrame(linhas)
pd.set_option("display.width", 140)
print(res.to_string(index=False) if not res.empty else "Nenhum pregao ausente.")
saida = config.RAIZ / "resultados" / "verificacao_calendario.csv"
res.to_csv(saida, index=False)
print(f"\nAtivos na base: {base['ativo'].nunique()} | pregoes no calendario: {len(calendario)}")
print(f"Casos de lacuna: {len(res)} | registros ausentes (ativo x data): {int(res['qtd_ausentes'].sum()) if len(res) else 0}")
print(f"Previsoes do periodo de teste afetadas: {int(res['previsao_afetada_no_teste'].sum()) if len(res) else 0} "
      f"de {len(prev)}")
print(f"Arquivo salvo em: {saida}")
