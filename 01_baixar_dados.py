"""
01_baixar_dados.py - ETAPA 3: CRIACAO DO BANCO DE DADOS (DOWNLOAD)
==================================================================

O que faz:
  1. Baixa as cotacoes diarias de cada ativo (abertura, maxima, minima, fechamento e
     volume) do Yahoo Finance, usando a biblioteca yfinance. Tambem baixa o VIX e o
     Ibovespa.
  2. Baixa a taxa diaria do CDI do Banco Central (serie 12 do SGS). O CDI e o
     rendimento de "ficar no caixa" no backtest.

Os downloads acontecem EM PARALELO com concurrent.futures:
enquanto um download espera a resposta do servidor, outro ja comeca. Como o tempo e
gasto esperando a rede (e nao calculando), varias "threads" ao mesmo tempo aceleram muito.

Arquivos gerados em dados/brutos/:
  precos/<TICKER>.csv        data, abertura, maxima, minima, fechamento, volume
  cdi.csv                    data, taxa diaria (em decimal: 0,0004 = 0,04% ao dia)
  relatorio_download.csv     o que foi baixado (ou o erro) de cada serie
  ultima_atualizacao.txt     data e hora do download (aparece no site)
"""
# ---------------------------------------------------------------------------
# IMPORTS: bibliotecas usadas neste script
# ---------------------------------------------------------------------------
import ssl                    # certificados de seguranca (HTTPS)
import sys                    # informacoes do sistema (ex.: se estamos no Windows)
import tempfile               # descobrir a pasta temporaria do sistema
import time                   # medir tempo e esperar (sleep)
from concurrent.futures import ThreadPoolExecutor, as_completed  # paralelismo
from datetime import datetime  # data e hora atuais
from pathlib import Path      # caminhos de arquivos

import certifi                # lista publica de certificados confiaveis
import pandas as pd           # tabelas (DataFrames)
import requests               # fazer requisicoes HTTP (usado para o Banco Central)
import yfinance as yf         # baixar dados do Yahoo Finance
from curl_cffi import requests as curl_requests  # cliente HTTP usado pelo yfinance

import config                 # o nosso config.py

# Pasta onde cada ativo vira um CSV
PASTA_PRECOS = config.DADOS_BRUTOS / "precos"

# Endereco da API do Banco Central. {ini} e {fim} sao "buracos" preenchidos depois
# com .format(). A API devolve um JSON: [{"data": "02/01/2018", "valor": "0.026444"}, ...]
URL_CDI = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
           "?formato=json&dataInicial={ini}&dataFinal={fim}")


# ---------------------------------------------------------------------------
# FUNCOES AUXILIARES
# ---------------------------------------------------------------------------
def com_tentativas(func, *args):
    """Executa func(*args) e, se der erro (ex.: internet instavel), tenta de novo.

    *args significa "qualquer quantidade de argumentos": com_tentativas(baixar_ticker, "PETR4")
    executa baixar_ticker("PETR4").
    """
    for tentativa in range(1, config.TENTATIVAS + 1):   # 1, 2, 3
        try:
            return func(*args)                          # deu certo: devolve o resultado
        except Exception as erro:  # noqa: BLE001      # deu errado: cai aqui
            if tentativa == config.TENTATIVAS:          # era a ultima tentativa?
                raise                                   # entao desiste e repassa o erro
            espera = 3 * tentativa                      # espera 3s, depois 6s...
            print(f"  falha ({erro}); nova tentativa em {espera}s")
            time.sleep(espera)


def pacote_certificados():
    """Monta um arquivo com todos os certificados de seguranca confiaveis.

    Por que existe: em redes de empresa, o trafego HTTPS costuma passar por um
    "inspetor" que usa um certificado proprio da empresa. O Windows conhece esse
    certificado, mas o yfinance nao, e o download falha com
    "self signed certificate in certificate chain" (foi o erro no PC da empresa).
    Solucao: juntar os certificados publicos (certifi) com os do Windows.
    """
    # 1) certificados publicos padrao
    partes = [Path(certifi.where()).read_text(encoding="utf-8")]
    # 2) certificados instalados no Windows (so existe essa funcao no Windows)
    if sys.platform == "win32":
        for loja in ("ROOT", "CA"):
            for cert, codificacao, _ in ssl.enum_certificates(loja):
                if codificacao == "x509_asn":
                    partes.append(ssl.DER_cert_to_PEM_cert(cert))  # converte formato
    # 3) grava tudo num arquivo na pasta temporaria (caminho sem acentos)
    destino = Path(tempfile.gettempdir()) / "tcc_previsao_certificados.pem"
    destino.write_text("\n".join(partes), encoding="utf-8")
    return str(destino)


# Executado uma unica vez quando o script comeca
CERTIFICADOS = pacote_certificados()


def nova_sessao():
    """Cria uma 'sessao' HTTP que usa os nossos certificados.

    impersonate="chrome" faz a requisicao parecer vir de um navegador Chrome
    (o Yahoo bloqueia alguns clientes "robos"). Cada download tem sua propria
    sessao porque threads diferentes nao devem dividir a mesma.
    """
    return curl_requests.Session(impersonate="chrome", verify=CERTIFICADOS)


def nome_arquivo(ticker):
    """'^VIX' -> 'VIX' (o simbolo ^ nao e bom em nome de arquivo)."""
    return ticker.replace("^", "")


# ---------------------------------------------------------------------------
# DOWNLOAD DE UM ATIVO
# ---------------------------------------------------------------------------
def baixar_ticker(ticker):
    """Baixa o historico diario de UM ativo e salva em precos/<TICKER>.csv."""
    # No Yahoo, ativos da B3 tem o sufixo ".SA" (PETR4 -> PETR4.SA).
    # Indices comecam com "^" (^VIX, ^BVSP) e nao levam sufixo.
    simbolo = ticker if ticker.startswith("^") else f"{ticker}.SA"

    # O parametro "end" do yfinance NAO inclui o proprio dia, entao somamos 1 dia.
    fim = (pd.Timestamp(config.DATA_FIM) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # auto_adjust=True: precos AJUSTADOS por dividendos e desdobramentos.
    # Sem isso, um desdobramento 1:2 apareceria como uma "queda" de 50% num dia.
    hist = yf.Ticker(simbolo, session=nova_sessao()).history(
        start=config.DATA_INICIO_DOWNLOAD, end=fim, auto_adjust=True, actions=False)

    if hist.empty:                                   # nada veio -> erro
        raise ValueError(f"nenhum dado para {simbolo}")

    # O Yahoo devolve datas com fuso horario; tiramos o fuso e a hora, deixando so a data.
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()

    # Montamos uma tabela com nomes de colunas em portugues.
    df = pd.DataFrame({
        "data": hist.index,
        "abertura": hist["Open"].values, "maxima": hist["High"].values,
        "minima": hist["Low"].values, "fechamento": hist["Close"].values,
        "volume": hist["Volume"].values,
    }).dropna(subset=["fechamento"]).drop_duplicates("data")  # sem fechamento vazio
                                                              # nem datas repetidas
    df.to_csv(PASTA_PRECOS / f"{nome_arquivo(ticker)}.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# DOWNLOAD DO CDI (BANCO CENTRAL)
# ---------------------------------------------------------------------------
def baixar_cdi_ano(ano):
    """Baixa o CDI de UM ano (a API do BC limita o tamanho de cada consulta)."""
    ini = max(pd.Timestamp(f"{ano}-01-01"), pd.Timestamp(config.DATA_INICIO_DOWNLOAD))
    fim = min(pd.Timestamp(f"{ano}-12-31"), pd.Timestamp(config.DATA_FIM))
    # A API espera datas no formato brasileiro dd/mm/aaaa
    url = URL_CDI.format(ini=ini.strftime("%d/%m/%Y"), fim=fim.strftime("%d/%m/%Y"))
    resp = requests.get(url, timeout=60, verify=CERTIFICADOS)  # faz a requisicao HTTP
    resp.raise_for_status()           # se o servidor respondeu com erro, gera excecao
    return pd.DataFrame(resp.json())  # transforma o JSON em tabela


def baixar_cdi():
    """Baixa o CDI de todos os anos (em paralelo) e salva em cdi.csv."""
    anos = range(pd.Timestamp(config.DATA_INICIO_DOWNLOAD).year,
                 pd.Timestamp(config.DATA_FIM).year + 1)          # 2017, 2018, ..., 2026
    # ex.map aplica a funcao a cada ano usando varias threads e devolve os resultados
    # NA MESMA ORDEM dos anos. "lambda a: ..." e uma funcao sem nome, de uma linha.
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
        partes = list(ex.map(lambda a: com_tentativas(baixar_cdi_ano, a), anos))
    cdi = pd.concat(partes, ignore_index=True)                  # junta os anos
    cdi["data"] = pd.to_datetime(cdi["data"], format="%d/%m/%Y")
    cdi["cdi_dia"] = pd.to_numeric(cdi["valor"]) / 100  # vem em %, passamos para decimal
    cdi = cdi[["data", "cdi_dia"]].drop_duplicates("data").sort_values("data")
    cdi.to_csv(config.DADOS_BRUTOS / "cdi.csv", index=False)
    print(f"CDI: {len(cdi)} dias")


def tipo_de(ticker):
    """Diz se o ticker e 'etf', 'acao', 'fii' ou 'auxiliar' (VIX, Ibovespa)."""
    for grupo, tipo in ((config.ETFS, "etf"), (config.ACOES, "acao"), (config.FIIS, "fii")):
        if ticker in grupo:
            return tipo
    return "auxiliar"


# ---------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    PASTA_PRECOS.mkdir(parents=True, exist_ok=True)   # cria a pasta se nao existir
    inicio = time.time()                              # marca o tempo inicial

    # Lista de tudo que sera baixado: os 10 ativos + VIX + Ibovespa
    tickers = (list(config.ETFS) + list(config.ACOES) + list(config.FIIS)
               + [config.INDICE_VIX, config.BENCHMARK])
    print(f"Baixando {len(tickers)} series com {config.MAX_WORKERS} downloads em paralelo...")

    cobertura = []   # aqui vamos guardando um resumo de cada download
    # ThreadPoolExecutor: um "gerente" com MAX_WORKERS "funcionarios" (threads).
    # O "with" garante que todas as threads terminem antes de seguir adiante.
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
        # ex.submit(...) entrega uma tarefa e devolve imediatamente um "futuro"
        # (um recibo de algo que vai terminar depois). Guardamos {futuro: ticker}.
        futuros = {ex.submit(com_tentativas, baixar_ticker, t): t for t in tickers}
        # as_completed entrega os futuros NA ORDEM EM QUE TERMINAM (nao na ordem pedida)
        for fut in as_completed(futuros):
            t = futuros[fut]
            try:
                df = fut.result()        # pega o resultado (ou o erro) daquela tarefa
                cobertura.append({"ativo": t, "tipo": tipo_de(t),
                                  "inicio": df["data"].min().date(),
                                  "fim": df["data"].max().date(), "pregoes": len(df)})
                print(f"  {t}: {len(df)} pregoes")
            except Exception as erro:  # noqa: BLE001
                # um ativo com erro NAO derruba o programa: anota e segue
                print(f"  {t}: ERRO {erro}")
                cobertura.append({"ativo": t, "tipo": tipo_de(t), "erro": str(erro)})

    try:
        baixar_cdi()
    except Exception as erro:  # noqa: BLE001
        print(f"CDI: ERRO {erro}")

    # Relatorio do que foi baixado e registro da hora da atualizacao
    rel = pd.DataFrame(cobertura).sort_values(["tipo", "ativo"])
    rel.to_csv(config.DADOS_BRUTOS / "relatorio_download.csv", index=False)
    (config.DADOS_BRUTOS / "ultima_atualizacao.txt").write_text(
        datetime.now().strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
    print("\nCobertura:\n" + rel.to_string(index=False))
    print(f"\nConcluido em {time.time() - inicio:.0f}s")


# Este "if" faz o main() rodar so quando o arquivo e executado diretamente
# ("python 01_baixar_dados.py"). Se outro arquivo (ou o notebook) fizer import dele,
# as funcoes ficam disponiveis mas o download NAO comeca sozinho.
if __name__ == "__main__":
    main()
