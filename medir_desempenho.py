"""
medir_desempenho.py - EVIDENCIA DE DESEMPENHO (capitulo de Desenvolvimento)
==========================================================================

Mede o ganho das duas formas de paralelismo usadas no sistema:
  1. Coleta: download SEQUENCIAL (1 thread) x CONCORRENTE (MAX_WORKERS threads).
     Tarefa limitada pela rede (I/O-bound) -> threads.
  2. Modelagem: walk-forward em 1 processo x em todos os nucleos (joblib).
     Tarefa limitada pela CPU (CPU-bound) -> processos.

Uso:  python medir_desempenho.py      (leva alguns minutos; rode depois do atualizar.py)
Saida: resultados/desempenho.csv e um resumo na tela (tire um print para a monografia).

Cuidados de medicao:
  - os downloads de teste vao para uma pasta TEMPORARIA, para nao sobrescrever os dados
    congelados do experimento;
  - a primeira rodada de download e descartada (aquecimento): nela a biblioteca obtem os
    cookies de acesso do Yahoo, o que distorceria a comparacao;
  - as rodadas seguintes alternam a ordem (sequencial/concorrente) e o resumo usa a mediana.
"""
import importlib
import os
import platform
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

import config

baixar = importlib.import_module("01_baixar_dados")
modelar = importlib.import_module("03_modelar")
# Esconde os avisos internos das bibliotecas. Fica DEPOIS dos imports porque o yfinance
# registra os proprios filtros de aviso ao ser importado.
warnings.simplefilter("ignore")
REPETICOES = 3

# downloads de teste numa pasta temporaria (nunca em dados/brutos)
baixar.PASTA_PRECOS = Path(tempfile.mkdtemp(prefix="tcc_desempenho_"))


def tempo_download(workers):
    tickers = (list(config.ETFS) + list(config.ACOES) + list(config.FIIS)
               + [config.INDICE_VIX, config.BENCHMARK])
    inicio = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda t: baixar.com_tentativas(baixar.baixar_ticker, t), tickers))
    return time.perf_counter() - inicio


def tempo_modelagem(n_jobs):
    dados = pd.read_parquet(config.DADOS_PROCESSADOS / "dataset.parquet")
    ultimo = pd.read_parquet(config.DADOS_PROCESSADOS / "ultimo_dia.parquet")
    ativos = sorted(dados["ativo"].unique())
    inicio = time.perf_counter()
    Parallel(n_jobs=n_jobs)(
        delayed(modelar.modelar_ativo)(dados[dados["ativo"] == a], ultimo[ultimo["ativo"] == a])
        for a in ativos)
    return time.perf_counter() - inicio


def main():
    print("Aquecimento (descartado)...")
    tempo_download(config.MAX_WORKERS)
    linhas = []
    seq = ("download sequencial (1 thread)", 1)
    con = (f"download concorrente ({config.MAX_WORKERS} threads)", config.MAX_WORKERS)
    for rep in range(1, REPETICOES + 1):
        ordem = [seq, con] if rep % 2 else [con, seq]     # alterna a ordem a cada rodada
        for nome, workers in ordem:
            s = tempo_download(workers)
            print(f"[{rep}] {nome}: {s:.2f} s")
            linhas.append({"etapa": nome, "repeticao": rep, "segundos": round(s, 3)})
    for nome, arg in [("modelagem em 1 processo", 1), (f"modelagem em {os.cpu_count()} nucleos", -1)]:
        s = tempo_modelagem(arg)
        print(f"{nome}: {s:.1f} s")
        linhas.append({"etapa": nome, "repeticao": 1, "segundos": round(s, 2)})

    df = pd.DataFrame(linhas)
    df["maquina"] = f"{platform.processor() or platform.machine()} | {os.cpu_count()} nucleos | {platform.system()} {platform.release()}"
    config.RESULTADOS.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTADOS / "desempenho.csv", index=False)

    mediana = df.groupby("etapa")["segundos"].median()
    print("\nResumo (mediana):")
    print(mediana.round(2).to_string())
    d_seq, d_con = mediana[seq[0]], mediana[con[0]]
    m_seq = mediana["modelagem em 1 processo"]
    m_par = mediana[f"modelagem em {os.cpu_count()} nucleos"]
    print(f"\nGanho na coleta:     {d_seq / d_con:.1f}x mais rapido")
    print(f"Ganho na modelagem:  {m_seq / m_par:.1f}x mais rapido")
    print(f"Maquina: {df['maquina'].iloc[0]}")


if __name__ == "__main__":
    main()
