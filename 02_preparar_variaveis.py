"""
02_preparar_variaveis.py - ETAPAS 2 E 4: TRATAMENTO E CALCULO DAS METRICAS
==========================================================================

Para cada ativo:
  1. le o CSV baixado pelo script 01 e trata os dados;
  2. aplica os filtros de entrada (historico completo e liquidez minima);
  3. calcula as 10 variaveis explicativas;
  4. cria o alvo: 1 se o fechamento ajustado do pregao seguinte for maior, 0 caso contrario;
  5. junta tudo numa tabela unica e salva.

Restricao temporal: a variavel do dia t usa apenas informacao disponivel ate o fechamento
de t (verificado por tests/test_pipeline.py e verificar_causalidade.py).

Variaveis explicativas (parametros usuais da analise tecnica):
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

Notacao: h = maxima, l = minima, c = fechamento, n = numero de periodos.

Arquivos gerados em dados/processados/:
  dataset.parquet     uma linha por (ativo, dia), com indicadores, alvo e retornos futuros
  ultimo_dia.parquet  variaveis do ultimo dia de cada ativo (previsao do pregao seguinte)
  ativos.csv          cadastro dos ativos avaliados, com o motivo de exclusao quando houver
"""
import numpy as np
import pandas as pd

import config


# ===========================================================================
# INDICADORES TECNICOS
# Janelas moveis (rolling, ewm) usam os n dias ate t, inclusive; nenhuma usa dias posteriores.
# ===========================================================================
def mme(x, n):
    """Media movel exponencial de n periodos (indefinida ate haver n observacoes)."""
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def cci(h, l, c, n=20, k=0.015):
    """Commodity Channel Index: afastamento do preco tipico em relacao a sua media de n dias,
    em unidades de desvio medio absoluto (constante 0,015 da definicao original)."""
    tp = (h + l + c) / 3                          # preco tipico
    media = tp.rolling(n).mean()
    desvio_medio = tp.rolling(n).apply(lambda v: np.mean(np.abs(v - v.mean())), raw=True)
    return (tp - media) / (k * desvio_medio)


def williams_r(h, l, c, n=14):
    """Williams %R: posicao do fechamento na faixa maxima-minima de n dias (0 a -100)."""
    hh, ll = h.rolling(n).max(), l.rolling(n).min()
    return -100 * (hh - c) / (hh - ll)


def smi(h, l, c, n=13, rapido=2, lento=25):
    """Stochastic Momentum Index: distancia do fechamento ao ponto medio da faixa de n dias,
    com dupla suavizacao exponencial (-100 a +100)."""
    hh, ll = h.rolling(n).max(), l.rolling(n).min()
    num = mme(mme(c - (hh + ll) / 2, lento), rapido)   # distancia ao meio da faixa
    den = mme(mme(hh - ll, lento), rapido)             # tamanho da faixa
    return 100 * num / (den / 2)


def trix(c, n=20):
    """TRIX: variacao percentual diaria da media exponencial tripla de n periodos."""
    tripla = mme(mme(mme(c, n), n), n)
    return 100 * tripla.pct_change()


def cmo(c, n=14):
    """Chande Momentum Oscillator: soma das altas menos soma das quedas em n dias,
    sobre a soma de ambas (-100 a +100)."""
    d = c.diff()
    alta = d.clip(lower=0).rolling(n).sum()
    baixa = (-d.clip(upper=0)).rolling(n).sum()
    return 100 * (alta - baixa) / (alta + baixa)


def dpo(c, n=10, deslocamento=6):
    """Detrended Price Oscillator, versao causal: preco de 'deslocamento' dias antes
    comparado com a media movel atual. A versao centrada (por exemplo, a do pacote TTR do R)
    desloca a media para o futuro e usaria informacao de t+1 a t+6."""
    return 100 * (c.shift(deslocamento) / c.rolling(n).mean() - 1)


def rsi(c, n=14):
    """Indice de Forca Relativa (usado so no conjunto 'ampliado', opcional)."""
    d = c.diff()
    ganho = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    perda = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + ganho / perda.replace(0, np.nan))


def calcular_variaveis(px, vix_var):
    """Calcula as variaveis de um ativo. vix_var ja vem alinhada as datas da B3."""
    h, l, c = px["maxima"], px["minima"], px["fechamento"]
    r = c.pct_change()
    df = pd.DataFrame(index=px.index)
    df["preco"] = c

    # ---- variaveis explicativas ----
    df["vix_var"] = vix_var.reindex(df.index)
    df["cci"] = cci(h, l, c)
    df["macd"] = 100 * (mme(c, 12) / mme(c, 26) - 1)   # MACD normalizado (PPO), em %
    df["williams_r"] = williams_r(h, l, c)
    df["stoch_smi"] = smi(h, l, c)
    df["trix"] = trix(c)
    mm20, dp20 = c.rolling(20).mean(), c.rolling(20).std()
    # %B de Bollinger: 0 na banda inferior (media - 2 desvios), 1 na superior
    df["bollinger_pctb"] = (c - (mm20 - 2 * dp20)) / (4 * dp20)
    df["cmo"] = cmo(c)
    df["dpo"] = dpo(c)
    df["roc"] = r                              # ROC de 1 periodo

    # ---- metricas de risco e retorno (conjunto "ampliado", desativado) ----
    for n in (5, 21, 63, 126):
        df[f"ret_{n}d"] = c.pct_change(n)
    for n in (21, 63, 252):
        df[f"vol_{n}d"] = r.rolling(n).std() * np.sqrt(252)    # volatilidade anualizada
    for n in (50, 200):
        df[f"dist_mm{n}"] = c / c.rolling(n).mean() - 1
    df["drawdown_252d"] = c / c.rolling(252, min_periods=1).max() - 1
    df["rsi_14"] = rsi(c)

    # ---- regime de volatilidade (analise de robustez) ----
    # mediana da volatilidade de 63 dias em todo o historico do ativo ate t-1
    mediana_passada = df["vol_63d"].expanding(min_periods=63).median().shift(1)
    df["regime_vol"] = np.where(df["vol_63d"] > mediana_passada, "alta", "baixa")
    df.loc[mediana_passada.isna(), "regime_vol"] = None   # sem historico suficiente
    return df


# ===========================================================================
# LEITURA, LIMPEZA E ALINHAMENTO DOS DADOS
# ===========================================================================
def ler_precos(ticker):
    """Le precos/<TICKER>.csv indexado por data (None se o arquivo nao existir)."""
    arq = config.DADOS_BRUTOS / "precos" / f"{ticker.replace('^', '')}.csv"
    if not arq.exists():
        return None
    return pd.read_csv(arq, parse_dates=["data"]).set_index("data").sort_index()


def tratar(px):
    """Tratamento dos precos: duplicatas, valores invalidos e saltos implausiveis."""
    px = px[~px.index.duplicated(keep="last")].copy()
    px = px[px["fechamento"] > 0]
    # abertura, maxima e minima invalidas (<= 0 ou ausentes) sao substituidas pelo fechamento
    for col in ("maxima", "minima", "abertura"):
        px[col] = px[col].where(px[col] > 0, px["fechamento"])
    # coerencia: a maxima nao fica abaixo da abertura/fechamento; a minima, nao acima
    px["maxima"] = px[["maxima", "fechamento", "abertura"]].max(axis=1)
    px["minima"] = px[["minima", "fechamento", "abertura"]].min(axis=1)
    # variacao diaria acima de 50% e tratada como erro de dado ou de ajuste
    salto = px["fechamento"].pct_change().abs() > 0.5
    if salto.any():
        print(f"    {salto.sum()} dia(s) com variacao > 50% removido(s)")
        px = px[~salto]
    return px


def variacao_vix():
    """Variacao percentual diaria do fechamento do VIX."""
    vix = ler_precos(config.INDICE_VIX)
    if vix is None:
        print("Aviso: sem VIX; a variavel vix_var ficara zerada.")
        return None
    var = vix["fechamento"].pct_change().dropna()
    return var.rename("vix_var")


def alinhar_vix(var, datas):
    """Associa a cada pregao t da B3 a variacao do VIX do ultimo dia americano anterior a t.

    O mercado americano fecha depois da B3: no fechamento de t, o VIX de t ainda nao e
    conhecido. Por isso o VIX entra com um dia de defasagem.
    """
    if var is None:
        return pd.Series(0.0, index=datas)
    esq = pd.DataFrame({"data": datas})                  # datas da B3
    dir_ = var.rename_axis("data_vix").reset_index()     # datas e valores do VIX
    # allow_exact_matches=False exclui a propria data (estritamente antes de t).
    m = pd.merge_asof(esq, dir_, left_on="data", right_on="data_vix",
                      allow_exact_matches=False)
    return pd.Series(m["vix_var"].values, index=datas)


def indice_cdi():
    """Indice acumulado do CDI; o rendimento entre duas datas e indice(fim)/indice(inicio) - 1."""
    arq = config.DADOS_BRUTOS / "cdi.csv"
    if not arq.exists():
        print("Aviso: sem CDI; o backtest vai considerar caixa com rendimento zero.")
        return None
    cdi = pd.read_csv(arq, parse_dates=["data"]).set_index("data")["cdi_dia"]
    return (1 + cdi).cumprod()


# ===========================================================================
# PROGRAMA PRINCIPAL
# ===========================================================================
def main():
    config.DADOS_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    h = config.HORIZONTE
    ini, fim = pd.Timestamp(config.DATA_INICIO), pd.Timestamp(config.DATA_FIM)
    tol = pd.Timedelta(days=config.TOLERANCIA_DIAS)
    var_vix, idx_cdi = variacao_vix(), indice_cdi()

    universo = ([(t, n, "etf") for t, n in config.ETFS.items()]
                + [(t, n, "acao") for t, n in config.ACOES.items()]
                + [(t, n, "fii") for t, n in config.FIIS.items()])

    tabelas, ultimos, cadastro = [], [], []
    for ativo, nome, tipo in universo:
        px = ler_precos(ativo)
        reg = {"ativo": ativo, "nome": nome, "tipo": tipo}
        if px is None:
            cadastro.append({**reg, "incluido": False, "motivo": "sem dados baixados"})
            continue
        px = tratar(px)

        # ---- FILTROS DE ENTRADA ----
        # Avaliados no periodo inteiro da amostra (fonte de vies de sobrevivencia).
        periodo = px.loc[ini:fim]
        # liquidez: mediana do volume financeiro diario (fechamento x volume)
        liquidez = (periodo["fechamento"] * periodo["volume"]).median()
        reg.update(primeiro_dia=px.index.min().date(), ultimo_dia=px.index.max().date(),
                   volume_financeiro_mediano=round(liquidez))
        if px.index.min() > ini + tol or px.index.max() < fim - tol:
            cadastro.append({**reg, "incluido": False, "motivo": "historico incompleto"})
            continue
        if liquidez < config.LIQUIDEZ_MINIMA:
            cadastro.append({**reg, "incluido": False, "motivo": "baixa liquidez"})
            continue

        print(f"  {ativo} ({tipo})")
        # ---- VARIAVEIS ----
        df = calcular_variaveis(px, alinhar_vix(var_vix, px.index))

        # ---- ALVO ----
        # ret_futuro = retorno de t ate t+h. ret_futuro e alvo nunca entram como variaveis
        # explicativas; servem para avaliar as previsoes e para o backtest.
        c = df["preco"]
        df["ret_futuro"] = c.shift(-h) / c - 1
        df["alvo"] = (df["ret_futuro"] > 0).astype(float)
        df.loc[df["ret_futuro"].isna(), "alvo"] = np.nan     # ultimo dia: alvo desconhecido

        # rendimento do CDI no mesmo intervalo (caixa no backtest)
        if idx_cdi is not None:
            cdi_no_ativo = idx_cdi.reindex(df.index, method="ffill")
            df["cdi_futuro"] = cdi_no_ativo.shift(-h) / cdi_no_ativo - 1
        else:
            df["cdi_futuro"] = 0.0

        df["ativo"], df["tipo"], df["nome"] = ativo, tipo, nome
        df["periodo"] = np.where(df.index >= pd.Timestamp(config.INICIO_RESERVADO),
                                 "reservado", "validacao")

        # ---- RECORTE FINAL ----
        df = df.loc[ini:fim]
        completo = df.dropna(subset=config.TODAS_VARIAVEIS + ["regime_vol"])
        ultimos.append(completo.tail(1))       # usado na previsao do pregao seguinte
        rotulado = completo.dropna(subset=["alvo"])
        tabelas.append(rotulado)
        cadastro.append({**reg, "incluido": True, "motivo": "",
                         "observacoes": len(rotulado),
                         "pct_alta": round(rotulado["alvo"].mean() * 100, 1)})

    # ---- SALVA ----
    dados = pd.concat(tabelas).rename_axis("data").reset_index()
    dados.to_parquet(config.DADOS_PROCESSADOS / "dataset.parquet", index=False)
    pd.concat(ultimos).rename_axis("data").reset_index() \
        .to_parquet(config.DADOS_PROCESSADOS / "ultimo_dia.parquet", index=False)
    cad = pd.DataFrame(cadastro)
    cad.to_csv(config.DADOS_PROCESSADOS / "ativos.csv", index=False)

    print(f"\nDataset: {len(dados):,} linhas, {dados['ativo'].nunique()} ativos incluidos")
    print(cad.to_string(index=False))


if __name__ == "__main__":
    main()
