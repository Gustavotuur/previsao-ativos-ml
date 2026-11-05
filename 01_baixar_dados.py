"""
01_baixar_dados.py - ETAPA 3: CRIACAO DO BANCO DE DADOS (DOWNLOAD)
==================================================================

1. Baixa as cotacoes diarias (abertura, maxima, minima, fechamento e volume) de cada
   ativo, do VIX e do Ibovespa no Yahoo Finance (yfinance), com precos ajustados.
2. Baixa a taxa diaria do CDI no Banco Central (SGS, serie 12), usada como
   rendimento do caixa no backtest.

Os downloads sao concorrentes (concurrent.futures.ThreadPoolExecutor): a tarefa e
limitada pela espera da rede, entao varias threads reduzem o tempo total.

Arquivos gerados em dados/brutos/:
  precos/<TICKER>.csv        data, abertura, maxima, minima, fechamento, volume
  cdi.csv                    data, taxa diaria em decimal
  relatorio_download.csv     cobertura (ou erro) de cada serie
  ultima_atualizacao.txt     data e hora do download
"""
import ssl
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import certifi
import pandas as pd
import requests
import yfinance as yf
from curl_cffi import requests as curl_requests

import config

PASTA_PRECOS = config.DADOS_BRUTOS / "precos"

# API do SGS/BCB; devolve [{"data": "02/01/2018", "valor": "0.026444"}, ...] (valor em %)
URL_CDI = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
           "?formato=json&dataInicial={ini}&dataFinal={fim}")


# ---------------------------------------------------------------------------
# FUNCOES AUXILIARES
# ---------------------------------------------------------------------------
def com_tentativas(func, *args):
    """Executa func(*args), repetindo em caso de falha (ate config.TENTATIVAS vezes)."""
    for tentativa in range(1, config.TENTATIVAS + 1):
        try:
            return func(*args)
        except Exception as erro:  # noqa: BLE001
            if tentativa == config.TENTATIVAS:
                raise
            espera = 3 * tentativa                      # 3 s, 6 s, ...
            print(f"  falha ({erro}); nova tentativa em {espera}s")
            time.sleep(espera)


def pacote_certificados():
    """Junta os certificados publicos (certifi) com os instalados no Windows.

    Em redes corporativas com inspecao de HTTPS, o certificado da empresa so existe
    no repositorio do Windows, e sem ele o download falha com
    "self signed certificate in certificate chain".
    """
    partes = [Path(certifi.where()).read_text(encoding="utf-8")]
    if sys.platform == "win32":
        for loja in ("ROOT", "CA"):
            for cert, codificacao, _ in ssl.enum_certificates(loja):
                if codificacao == "x509_asn":
                    partes.append(ssl.DER_cert_to_PEM_cert(cert))
    # pasta temporaria do sistema (caminho sem acentos)
    destino = Path(tempfile.gettempdir()) / "tcc_previsao_certificados.pem"
    destino.write_text("\n".join(partes), encoding="utf-8")
    return str(destino)


CERTIFICADOS = pacote_certificados()


def nova_sessao():
    """Sessao HTTP com os certificados acima, uma por download (nao compartilhada
    entre threads). impersonate="chrome" evita o bloqueio do Yahoo a clientes automatizados."""
    return curl_requests.Session(impersonate="chrome", verify=CERTIFICADOS)


def nome_arquivo(ticker):
    """'^VIX' -> 'VIX'."""
    return ticker.replace("^", "")


# ---------------------------------------------------------------------------
# DOWNLOAD DE UM ATIVO
# ---------------------------------------------------------------------------
def baixar_ticker(ticker):
    """Baixa o historico diario de um ativo e salva em precos/<TICKER>.csv."""
    # Ativos da B3 levam o sufixo ".SA" no Yahoo; indices (^VIX, ^BVSP) nao.
    simbolo = ticker if ticker.startswith("^") else f"{ticker}.SA"

    # "end" do yfinance e exclusivo: soma-se 1 dia para incluir DATA_FIM.
    fim = (pd.Timestamp(config.DATA_FIM) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # auto_adjust=True: precos ajustados por proventos e desdobramentos (sem o ajuste,
    # um desdobramento apareceria como uma queda artificial no retorno do dia).
    hist = yf.Ticker(simbolo, session=nova_sessao()).history(
        start=config.DATA_INICIO_DOWNLOAD, end=fim, auto_adjust=True, actions=False)

    if hist.empty:
        raise ValueError(f"nenhum dado para {simbolo}")

    # remove fuso horario e hora, mantendo so a data do pregao
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()

    df = pd.DataFrame({
        "data": hist.index,
        "abertura": hist["Open"].values, "maxima": hist["High"].values,
        "minima": hist["Low"].values, "fechamento": hist["Close"].values,
        "volume": hist["Volume"].values,
    }).dropna(subset=["fechamento"]).drop_duplicates("data")
    df.to_csv(PASTA_PRECOS / f"{nome_arquivo(ticker)}.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# DOWNLOAD DO CDI (BANCO CENTRAL)
# ---------------------------------------------------------------------------
def baixar_cdi_ano(ano):
    """Baixa o CDI de um ano (a API do BCB limita o intervalo de cada consulta)."""
    ini = max(pd.Timestamp(f"{ano}-01-01"), pd.Timestamp(config.DATA_INICIO_DOWNLOAD))
    fim = min(pd.Timestamp(f"{ano}-12-31"), pd.Timestamp(config.DATA_FIM))
    url = URL_CDI.format(ini=ini.strftime("%d/%m/%Y"), fim=fim.strftime("%d/%m/%Y"))
    resp = requests.get(url, timeout=60, verify=CERTIFICADOS)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def baixar_cdi():
    """Baixa o CDI de todos os anos (concorrentemente) e salva em cdi.csv."""
    anos = range(pd.Timestamp(config.DATA_INICIO_DOWNLOAD).year,
                 pd.Timestamp(config.DATA_FIM).year + 1)
    # ex.map devolve os resultados na ordem dos anos
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
        partes = list(ex.map(lambda a: com_tentativas(baixar_cdi_ano, a), anos))
    cdi = pd.concat(partes, ignore_index=True)
    cdi["data"] = pd.to_datetime(cdi["data"], format="%d/%m/%Y")
    cdi["cdi_dia"] = pd.to_numeric(cdi["valor"]) / 100  # % -> decimal
    cdi = cdi[["data", "cdi_dia"]].drop_duplicates("data").sort_values("data")
    cdi.to_csv(config.DADOS_BRUTOS / "cdi.csv", index=False)
    print(f"CDI: {len(cdi)} dias")


def tipo_de(ticker):
    """'etf', 'acao', 'fii' ou 'auxiliar' (VIX, Ibovespa)."""
    for grupo, tipo in ((config.ETFS, "etf"), (config.ACOES, "acao"), (config.FIIS, "fii")):
        if ticker in grupo:
            return tipo
    return "auxiliar"


# ---------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    PASTA_PRECOS.mkdir(parents=True, exist_ok=True)
    inicio = time.time()

    # ativos da amostra + VIX + Ibovespa
    tickers = (list(config.ETFS) + list(config.ACOES) + list(config.FIIS)
               + [config.INDICE_VIX, config.BENCHMARK])
    print(f"Baixando {len(tickers)} series com {config.MAX_WORKERS} downloads em paralelo...")

    cobertura = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
        futuros = {ex.submit(com_tentativas, baixar_ticker, t): t for t in tickers}
        for fut in as_completed(futuros):        # na ordem em que terminam
            t = futuros[fut]
            try:
                df = fut.result()
                cobertura.append({"ativo": t, "tipo": tipo_de(t),
                                  "inicio": df["data"].min().date(),
                                  "fim": df["data"].max().date(), "pregoes": len(df)})
                print(f"  {t}: {len(df)} pregoes")
            except Exception as erro:  # noqa: BLE001
                # a falha de uma serie e registrada sem interromper as demais
                print(f"  {t}: ERRO {erro}")
                cobertura.append({"ativo": t, "tipo": tipo_de(t), "erro": str(erro)})

    try:
        baixar_cdi()
    except Exception as erro:  # noqa: BLE001
        print(f"CDI: ERRO {erro}")

    rel = pd.DataFrame(cobertura).sort_values(["tipo", "ativo"])
    rel.to_csv(config.DADOS_BRUTOS / "relatorio_download.csv", index=False)
    (config.DADOS_BRUTOS / "ultima_atualizacao.txt").write_text(
        datetime.now().strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
    print("\nCobertura:\n" + rel.to_string(index=False))
    print(f"\nConcluido em {time.time() - inicio:.0f}s")


if __name__ == "__main__":
    main()
