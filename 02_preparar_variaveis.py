"""
02_preparar_variaveis.py - ETAPAS 2 E 4: TRATAMENTO E CALCULO DAS METRICAS
==========================================================================

O que faz, para cada ativo:
  1. le o CSV baixado pelo script 01 e LIMPA os dados;
  2. aplica os FILTROS de entrada (historico completo e liquidez minima);
  3. calcula os 10 INDICADORES tecnicos para cada dia;
  4. cria o ALVO (o que o modelo tenta prever: o preco sobe ou cai amanha?);
  5. junta tudo numa tabela unica e salva.

REGRA DE OURO: o indicador do dia t so pode usar informacao disponivel ate o
fechamento do dia t. Usar qualquer dado de t+1 em diante seria "olhar o futuro"
(vazamento de dados) e deixaria os resultados falsamente bons.

Os 10 indicadores tecnicos (parametros classicos da literatura de analise tecnica):
  vix_var         variacao do VIX do ultimo pregao americano ANTERIOR a t
  cci             Commodity Channel Index (20 periodos, constante 0,015)
  macd            MACD (12, 26), em %: 100 * (MME12 / MME26 - 1)
  williams_r      Williams %R (14 periodos), de -100 a 0
  stoch_smi       Stochastic Momentum Index (13, 2, 25), versao suavizada do estocastico
  trix            TRIX - media exponencial tripla (20 periodos), variacao em %
  bollinger_pctb  posicao dentro das Bandas de Bollinger (20 periodos, 2 desvios)
  cmo             Chande Momentum Oscillator (14 periodos)
  dpo             Detrended Price Oscillator (10 periodos, deslocamento 6), versao causal
  roc             Rate of Change de 1 periodo (o retorno do dia)

Notacao usada nas funcoes: h = maxima (high), l = minima (low), c = fechamento (close),
n = numero de periodos (dias). Uma "Series" do pandas e uma coluna com datas no indice.

Arquivos gerados em dados/processados/:
  dataset.parquet     uma linha por (ativo, dia), com indicadores, alvo e retornos futuros
  ultimo_dia.parquet  os indicadores do ultimo dia de cada ativo (previsao "de hoje" no site)
  ativos.csv          cada ativo avaliado: se entrou no estudo e, se nao, por que
"""
import numpy as np     # calculo numerico
import pandas as pd    # tabelas

import config


# ===========================================================================
# INDICADORES TECNICOS
# Cada funcao recebe colunas de preco e devolve uma coluna com o indicador dia a dia.
# Operacoes "rolling(n)" olham a janela dos ULTIMOS n dias (incluindo o dia atual) -
# nunca dias futuros. Por isso todos os indicadores abaixo sao causais.
# ===========================================================================
def mme(x, n):
    """Media Movel Exponencial de n periodos.

    Diferente da media simples, da mais peso aos dias recentes. min_periods=n faz o
    resultado ficar vazio (NaN) ate existirem n dias de dados.
    """
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def cci(h, l, c, n=20, k=0.015):
    """CCI: o quanto o "preco tipico" de hoje se afasta da sua media de n dias.

    Valores acima de +100 sugerem preco esticado para cima; abaixo de -100, para baixo.
    A constante 0,015 (definicao original do CCI) faz cerca de 70 a 80% dos valores ficarem entre -100 e +100.
    """
    tp = (h + l + c) / 3                          # preco tipico do dia
    media = tp.rolling(n).mean()                  # media dos ultimos n precos tipicos
    # desvio medio: media das distancias absolutas de cada valor ate a media da janela
    desvio_medio = tp.rolling(n).apply(lambda v: np.mean(np.abs(v - v.mean())), raw=True)
    return (tp - media) / (k * desvio_medio)


def williams_r(h, l, c, n=14):
    """Williams %R: onde o fechamento esta entre a maxima e a minima dos ultimos n dias.

    0 = fechou na maxima do periodo; -100 = fechou na minima.
    """
    hh, ll = h.rolling(n).max(), l.rolling(n).min()   # maior maxima e menor minima
    return -100 * (hh - c) / (hh - ll)


def smi(h, l, c, n=13, rapido=2, lento=25):
    """Stochastic Momentum Index: versao suavizada do oscilador estocastico.

    Mede a distancia do fechamento ate o MEIO da faixa (maxima-minima) dos ultimos n
    dias, suavizada duas vezes por medias exponenciais. Vai de -100 a +100.
    """
    hh, ll = h.rolling(n).max(), l.rolling(n).min()
    num = mme(mme(c - (hh + ll) / 2, lento), rapido)   # distancia ao meio da faixa
    den = mme(mme(hh - ll, lento), rapido)             # tamanho da faixa
    return 100 * num / (den / 2)


def trix(c, n=20):
    """TRIX: variacao % de uma media exponencial aplicada 3 vezes.

    Suavizar tres vezes remove o "ruido" do dia a dia e deixa so a tendencia.
    """
    tripla = mme(mme(mme(c, n), n), n)
    return 100 * tripla.pct_change()    # pct_change() = variacao em relacao ao dia anterior


def cmo(c, n=14):
    """Chande Momentum Oscillator: forca das altas contra a forca das quedas em n dias.

    +100 = so subiu no periodo; -100 = so caiu.
    """
    d = c.diff()                                   # variacao do preco em relacao a ontem
    alta = d.clip(lower=0).rolling(n).sum()        # soma so das variacoes positivas
    baixa = (-d.clip(upper=0)).rolling(n).sum()    # soma so das negativas (em modulo)
    return 100 * (alta - baixa) / (alta + baixa)


def dpo(c, n=10, deslocamento=6):
    """Detrended Price Oscillator: o preco "sem a tendencia", para enxergar ciclos.

    VERSAO CAUSAL: compara o preco de 'deslocamento' dias atras com a media movel
    atual. A versao "centrada" de alguns pacotes (como o TTR do R) desloca a media
    para o FUTURO e usaria informacao de t+1 a t+6, ou seja, olharia o futuro.
    """
    return 100 * (c.shift(deslocamento) / c.rolling(n).mean() - 1)


def rsi(c, n=14):
    """Indice de Forca Relativa (usado so no conjunto 'ampliado', opcional)."""
    d = c.diff()
    ganho = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    perda = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + ganho / perda.replace(0, np.nan))


def calcular_variaveis(px, vix_var):
    """Recebe a tabela de precos de um ativo e devolve uma tabela com todos os indicadores.

    px      = DataFrame com colunas abertura, maxima, minima, fechamento, volume
    vix_var = Series com a variacao do VIX ja alinhada as datas da B3
    """
    h, l, c = px["maxima"], px["minima"], px["fechamento"]
    r = c.pct_change()                         # retorno diario
    df = pd.DataFrame(index=px.index)          # tabela vazia com as mesmas datas
    df["preco"] = c

    # ---- os 10 indicadores tecnicos ----
    df["vix_var"] = vix_var.reindex(df.index)  # encaixa a serie do VIX nas datas do ativo
    df["cci"] = cci(h, l, c)
    df["macd"] = 100 * (mme(c, 12) / mme(c, 26) - 1)   # media rapida vs lenta, em %
    df["williams_r"] = williams_r(h, l, c)
    df["stoch_smi"] = smi(h, l, c)
    df["trix"] = trix(c)
    mm20, dp20 = c.rolling(20).mean(), c.rolling(20).std()   # media e desvio de 20 dias
    # %B de Bollinger: 0 = na banda inferior (media - 2 desvios), 1 = na banda superior
    df["bollinger_pctb"] = (c - (mm20 - 2 * dp20)) / (4 * dp20)
    df["cmo"] = cmo(c)
    df["dpo"] = dpo(c)
    df["roc"] = r                              # retorno do proprio dia

    # ---- metricas de risco e retorno (so usadas se o conjunto "ampliado" for ligado) ----
    for n in (5, 21, 63, 126):
        df[f"ret_{n}d"] = c.pct_change(n)                      # retorno acumulado em n dias
    for n in (21, 63, 252):
        df[f"vol_{n}d"] = r.rolling(n).std() * np.sqrt(252)    # volatilidade anualizada
    for n in (50, 200):
        df[f"dist_mm{n}"] = c / c.rolling(n).mean() - 1        # distancia a media movel
    df["drawdown_252d"] = c / c.rolling(252, min_periods=1).max() - 1   # queda desde o topo
    df["rsi_14"] = rsi(c)

    # ---- regime de volatilidade (usado na analise de robustez) ----
    # mediana da volatilidade de TODO o passado do ativo ate ONTEM (.shift(1)):
    mediana_passada = df["vol_63d"].expanding(min_periods=63).median().shift(1)
    # np.where(condicao, valor_se_sim, valor_se_nao), aplicado linha a linha
    df["regime_vol"] = np.where(df["vol_63d"] > mediana_passada, "alta", "baixa")
    df.loc[mediana_passada.isna(), "regime_vol"] = None   # sem historico suficiente
    return df


# ===========================================================================
# LEITURA, LIMPEZA E ALINHAMENTO DOS DADOS
# ===========================================================================
def ler_precos(ticker):
    """Le precos/<TICKER>.csv e devolve uma tabela com a data como indice (ou None)."""
    arq = config.DADOS_BRUTOS / "precos" / f"{ticker.replace('^', '')}.csv"
    if not arq.exists():
        return None
    return pd.read_csv(arq, parse_dates=["data"]).set_index("data").sort_index()


def tratar(px):
    """Limpeza: remove problemas comuns em dados de mercado."""
    px = px[~px.index.duplicated(keep="last")].copy()   # datas repetidas: fica a ultima
    px = px[px["fechamento"] > 0]                       # precos zerados ou negativos: fora
    # maxima/minima/abertura invalidas (<= 0) sao substituidas pelo fechamento
    for col in ("maxima", "minima", "abertura"):
        px[col] = px[col].where(px[col] > 0, px["fechamento"])
    # garante coerencia: a maxima e o maior valor do dia e a minima, o menor
    px["maxima"] = px[["maxima", "fechamento", "abertura"]].max(axis=1)
    px["minima"] = px[["minima", "fechamento", "abertura"]].min(axis=1)
    # variacao acima de 50% num dia quase sempre e erro de dado ou de ajuste
    salto = px["fechamento"].pct_change().abs() > 0.5
    if salto.any():
        print(f"    {salto.sum()} dia(s) com variacao > 50% removido(s)")
        px = px[~salto]                                  # "~" significa "nao"
    return px


def variacao_vix():
    """Variacao diaria do VIX: (VIX de hoje - VIX de ontem) / VIX de ontem."""
    vix = ler_precos(config.INDICE_VIX)
    if vix is None:
        print("Aviso: sem VIX; a variavel vix_var ficara zerada.")
        return None
    var = vix["fechamento"].pct_change().dropna()
    return var.rename("vix_var")


def alinhar_vix(var, datas):
    """Para cada dia t da B3, pega a variacao do VIX do ultimo dia americano ANTES de t.

    Por que "antes": o mercado americano fecha DEPOIS da B3. No fechamento de hoje
    na B3, o VIX de hoje ainda nao existe; usar o do mesmo dia seria um pequeno
    vazamento de informacao. Por isso o VIX entra com um dia de defasagem.
    """
    if var is None:
        return pd.Series(0.0, index=datas)
    esq = pd.DataFrame({"data": datas})                  # datas da B3
    dir_ = var.rename_axis("data_vix").reset_index()     # datas e valores do VIX
    # merge_asof: para cada data da esquerda, busca a data mais proxima ANTERIOR na direita.
    # allow_exact_matches=False exclui a propria data (estritamente antes de t).
    m = pd.merge_asof(esq, dir_, left_on="data", right_on="data_vix",
                      allow_exact_matches=False)
    return pd.Series(m["vix_var"].values, index=datas)


def indice_cdi():
    """Transforma a taxa diaria do CDI num indice acumulado (1,00 -> 1,0004 -> ...).

    Com o indice, o rendimento do CDI entre duas datas e simplesmente
    indice(fim) / indice(inicio) - 1.
    """
    arq = config.DADOS_BRUTOS / "cdi.csv"
    if not arq.exists():
        print("Aviso: sem CDI; o backtest vai considerar caixa com rendimento zero.")
        return None
    cdi = pd.read_csv(arq, parse_dates=["data"]).set_index("data")["cdi_dia"]
    return (1 + cdi).cumprod()    # produto acumulado: juros compostos


# ===========================================================================
# PROGRAMA PRINCIPAL
# ===========================================================================
def main():
    config.DADOS_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    h = config.HORIZONTE
    ini, fim = pd.Timestamp(config.DATA_INICIO), pd.Timestamp(config.DATA_FIM)
    tol = pd.Timedelta(days=config.TOLERANCIA_DIAS)
    var_vix, idx_cdi = variacao_vix(), indice_cdi()

    # Lista de (ticker, nome, tipo) de todos os ativos candidatos
    universo = ([(t, n, "etf") for t, n in config.ETFS.items()]
                + [(t, n, "acao") for t, n in config.ACOES.items()]
                + [(t, n, "fii") for t, n in config.FIIS.items()])

    tabelas, ultimos, cadastro = [], [], []
    for ativo, nome, tipo in universo:
        px = ler_precos(ativo)
        reg = {"ativo": ativo, "nome": nome, "tipo": tipo}   # registro para ativos.csv
        if px is None:
            cadastro.append({**reg, "incluido": False, "motivo": "sem dados baixados"})
            continue                                         # pula para o proximo ativo
        px = tratar(px)

        # ---- FILTROS DE ENTRADA ----
        periodo = px.loc[ini:fim]
        # volume financeiro diario = preco x quantidade negociada; usamos a mediana
        liquidez = (periodo["fechamento"] * periodo["volume"]).median()
        # {**reg, ...} cria um novo dicionario copiando reg e acrescentando campos
        reg.update(primeiro_dia=px.index.min().date(), ultimo_dia=px.index.max().date(),
                   volume_financeiro_mediano=round(liquidez))
        if px.index.min() > ini + tol or px.index.max() < fim - tol:
            cadastro.append({**reg, "incluido": False, "motivo": "historico incompleto"})
            continue
        if liquidez < config.LIQUIDEZ_MINIMA:
            cadastro.append({**reg, "incluido": False, "motivo": "baixa liquidez"})
            continue

        print(f"  {ativo} ({tipo})")
        # ---- INDICADORES ----
        df = calcular_variaveis(px, alinhar_vix(var_vix, px.index))

        # ---- ALVO ----
        # shift(-h) "puxa" o preco de h dias A FRENTE para a linha de hoje.
        # ret_futuro = quanto o ativo vai render de hoje ate amanha (h = 1).
        # ATENCAO: ret_futuro e o alvo NUNCA entram como variaveis do modelo;
        # servem so para dizer a resposta certa e para calcular o backtest.
        c = df["preco"]
        df["ret_futuro"] = c.shift(-h) / c - 1
        df["alvo"] = (df["ret_futuro"] > 0).astype(float)    # 1.0 se subiu, 0.0 se nao
        df.loc[df["ret_futuro"].isna(), "alvo"] = np.nan     # ultimo dia: ainda sem resposta

        # rendimento do CDI no mesmo intervalo (o que se ganharia ficando no caixa)
        if idx_cdi is not None:
            cdi_no_ativo = idx_cdi.reindex(df.index, method="ffill")  # ffill: repete o
            df["cdi_futuro"] = cdi_no_ativo.shift(-h) / cdi_no_ativo - 1  # ultimo valor
        else:
            df["cdi_futuro"] = 0.0

        # colunas de identificacao
        df["ativo"], df["tipo"], df["nome"] = ativo, tipo, nome
        df["periodo"] = np.where(df.index >= pd.Timestamp(config.INICIO_RESERVADO),
                                 "reservado", "validacao")

        # ---- RECORTE FINAL ----
        df = df.loc[ini:fim]                                         # so 2018 em diante
        completo = df.dropna(subset=config.TODAS_VARIAVEIS + ["regime_vol"])  # sem lacunas
        ultimos.append(completo.tail(1))       # ultimo dia: guardado para a previsao "de hoje"
        rotulado = completo.dropna(subset=["alvo"])    # so dias que ja tem resposta
        tabelas.append(rotulado)
        cadastro.append({**reg, "incluido": True, "motivo": "",
                         "observacoes": len(rotulado),
                         "pct_alta": round(rotulado["alvo"].mean() * 100, 1)})

    # ---- SALVA ----
    # pd.concat empilha as tabelas de todos os ativos numa so.
    # .rename_axis("data").reset_index() transforma o indice (datas) em coluna normal.
    dados = pd.concat(tabelas).rename_axis("data").reset_index()
    # Parquet: formato binario, mais rapido e menor que CSV, e preserva os tipos das colunas
    dados.to_parquet(config.DADOS_PROCESSADOS / "dataset.parquet", index=False)
    pd.concat(ultimos).rename_axis("data").reset_index() \
        .to_parquet(config.DADOS_PROCESSADOS / "ultimo_dia.parquet", index=False)
    cad = pd.DataFrame(cadastro)
    cad.to_csv(config.DADOS_PROCESSADOS / "ativos.csv", index=False)

    print(f"\nDataset: {len(dados):,} linhas, {dados['ativo'].nunique()} ativos incluidos")
    print(cad.to_string(index=False))


if __name__ == "__main__":
    main()
