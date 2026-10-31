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
"""
import importlib
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from joblib import Parallel, delayed

import config

baixar = importlib.import_module("01_baixar_dados")
modelar = importlib.import_module("03_modelar")
REPETICOES = 2


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
    baixar.PASTA_PRECOS.mkdir(parents=True, exist_ok=True)
    linhas = []
    for rep in range(1, REPETICOES + 1):
        for nome, f, arg in [("download sequencial (1 thread)", tempo_download, 1),
                             (f"download concorrente ({config.MAX_WORKERS} threads)", tempo_download,
                              config.MAX_WORKERS)]:
            s = f(arg)
            print(f"[{rep}] {nome}: {s:.1f} s")
            linhas.append({"etapa": nome, "repeticao": rep, "segundos": round(s, 2)})
    for nome, arg in [("modelagem em 1 processo", 1), (f"modelagem em {os.cpu_count()} nucleos", -1)]:
        s = tempo_modelagem(arg)
        print(f"{nome}: {s:.1f} s")
        linhas.append({"etapa": nome, "repeticao": 1, "segundos": round(s, 2)})

    df = pd.DataFrame(linhas)
    df["maquina"] = f"{platform.processor() or platform.machine()} | {os.cpu_count()} nucleos | {platform.system()} {platform.release()}"
    config.RESULTADOS.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTADOS / "desempenho.csv", index=False)

    media = df.groupby("etapa", sort=False)["segundos"].mean()
    print("\nResumo (media):")
    print(media.round(1).to_string())
    d_seq, d_con = media.iloc[0], media.iloc[1]
    m_seq, m_par = media.iloc[2], media.iloc[3]
    print(f"\nGanho na coleta:     {d_seq / d_con:.1f}x mais rapido")
    print(f"Ganho na modelagem:  {m_seq / m_par:.1f}x mais rapido")
    print(f"Maquina: {df['maquina'].iloc[0]}")


if __name__ == "__main__":
    main()
