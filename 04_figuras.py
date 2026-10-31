"""Gera as figuras da monografia em PNG e PDF.

Figuras:
    fig1_acuracia_por_ativo      acurácia de cada modelo em cada ativo
    fig2_acuracia_por_ano        acurácia por ano, com a linha de base
    fig3_acuracia_por_regime     acurácia em volatilidade baixa e alta
    fig4_importancia_variaveis   importância dos indicadores na Random Forest
    fig5_backtest_<ATIVO>        patrimônio acumulado das estratégias
    fig6_correlacao_<ATIVO>      correlação entre os indicadores
    fig7_painel_<ATIVO>          painel-resumo (tearsheet) de um ativo
"""

import matplotlib

matplotlib.use("Agg")   # gera arquivos sem abrir janela

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

import config


CONJUNTO = "ampliado" if "ampliado" in config.CONJUNTOS else "tecnicos"

# Nome nos resultados, nome apresentado, cor, marcador e linha.
MODELOS = {
    "logistica": (
        "Regressao Logistica", "Regressão Logística",
        "#275D80", "o", "-"
    ),
    "random_forest": (
        "Random Forest", "Random Forest",
        "#B56B35", "s", "--"
    ),
    "rede_neural": (
        "Rede Neural (MLP)", "Rede Neural (MLP)",
        "#39816B", "^", "-."
    ),
}

ESTILOS = {
    MODELOS[chave][0]: MODELOS[chave][1:]
    for chave in config.MODELOS
}
ESTILOS["Sempre alta"] = (
    "Sempre alta", "#717171", "D", ":"
)

COR_MANTER = "#333333"   # comprar e manter
COR_CDI = "#858585"

NOMES_VARIAVEIS = {
    "vix_var": "Variação do VIX",
    "cci": "CCI",
    "macd": "MACD normalizado",
    "williams_r": "Williams %R",
    "stoch_smi": "SMI",
    "trix": "TRIX",
    "bollinger_pctb": "Bollinger %B",
    "cmo": "CMO",
    "dpo": "DPO normalizado",
    "roc": "ROC",
    "rsi_14": "RSI (14)",
    "drawdown_252d": "Drawdown (252 dias)",
}

MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
         "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

sns.set_theme(
    context="paper",
    style="ticks",
    font="DejaVu Sans",
    rc={
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
)


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def nome_variavel(nome):
    return NOMES_VARIAVEIS.get(nome, nome.replace("_", " "))


def filtrar(tabela):
    """Mantém o conjunto de variáveis escolhido e a linha de base sempre alta."""
    selecionado = (
        tabela["conjunto"].eq(CONJUNTO)
        | tabela["modelo"].eq("Sempre alta")
    )
    return tabela.loc[
        selecionado & tabela["modelo"].isin(ESTILOS)
    ].copy()


def preparar_eixo(ax, grade="y"):
    ax.grid(False)
    ax.grid(axis=grade, color="#E5E5E5", linewidth=0.5)
    sns.despine(ax=ax)


def formato_pct(casas=1):
    return FuncFormatter(
        lambda valor, _: f"{100 * valor:.{casas}f}%".replace(".", ",").replace("-", "−")
    )


def porcentagem(ax, eixo="y", casas=1):
    alvo = ax.xaxis if eixo == "x" else ax.yaxis
    alvo.set_major_formatter(formato_pct(casas))
    alvo.set_major_locator(MaxNLocator(nbins=6))


def legenda(ax, colunas=2):
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=colunas,
        borderaxespad=0,
        handlelength=2.3,
        columnspacing=1.4,
    )


def salvar(fig, nome, ajustar=True):
    config.FIGURAS.mkdir(parents=True, exist_ok=True)
    if ajustar:
        fig.tight_layout(pad=1.0)

    for extensao in ("png", "pdf"):
        destino = config.FIGURAS / f"{nome}.{extensao}"
        opcoes = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0.08}
        try:
            fig.savefig(destino, **opcoes)
        except OSError as erro:
            # No Windows, um arquivo aberto em outro programa (visualizador de imagens,
            # VS Code, Word) fica bloqueado para escrita. Em vez de parar tudo, grava
            # com outro nome e avisa.
            alternativo = destino.with_name(f"{nome}_novo.{extensao}")
            print(f"  AVISO: não consegui gravar {destino.name} ({erro}).")
            print(f"         Feche o arquivo se ele estiver aberto. Gravando como {alternativo.name}.")
            fig.savefig(alternativo, **opcoes)

    plt.close(fig)
    print(f"  {nome}: PNG e PDF")


def modelos_presentes(tabela):
    return [modelo for modelo in ESTILOS if modelo in tabela.columns]


def titulo_painel(ax, texto):
    """Título pequeno, alinhado à esquerda, como nos relatórios de estratégia."""
    ax.set_title(texto, loc="left", fontsize=9, fontweight="bold", pad=6)


# ---------------------------------------------------------------------------
# Figuras 1 a 6
# ---------------------------------------------------------------------------
def fig_acuracia_ativo():
    dados = filtrar(
        pd.read_csv(config.RESULTADOS / "metricas_por_ativo.csv")
    )
    tabela = dados.pivot(
        index="ativo", columns="modelo", values="acuracia"
    ).sort_index()

    modelos = modelos_presentes(tabela)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    posicoes = np.arange(len(tabela))
    deslocamentos = np.linspace(-0.18, 0.18, len(modelos))

    for modelo, deslocamento in zip(modelos, deslocamentos):
        nome, cor, marcador, _ = ESTILOS[modelo]
        ax.scatter(
            tabela[modelo],
            posicoes + deslocamento,
            color=cor,
            marker=marcador,
            s=25,
            label=nome,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    ax.axvline(0.5, color="#888888", linestyle=":", linewidth=0.8)
    ax.set_yticks(posicoes, tabela.index)
    ax.invert_yaxis()
    ax.set_xlabel("Acurácia fora da amostra")
    ax.set_ylabel("")
    ax.margins(x=0.12)

    porcentagem(ax, "x")
    preparar_eixo(ax, "x")
    legenda(ax)
    salvar(fig, "fig1_acuracia_por_ativo")


def fig_acuracia_ano():
    dados = filtrar(
        pd.read_csv(config.RESULTADOS / "metricas_por_ano.csv")
    )
    tabela = dados.pivot(
        index="ano", columns="modelo", values="acuracia"
    ).sort_index()

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    for modelo in modelos_presentes(tabela):
        nome, cor, marcador, linha = ESTILOS[modelo]
        ax.plot(
            tabela.index,
            tabela[modelo],
            label=nome,
            color=cor,
            marker=marcador,
            linestyle=linha,
            markersize=4,
            linewidth=1.2,
        )

    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=0.8)
    ax.set_xticks(tabela.index)
    ax.set_xlabel("Ano")
    ax.set_ylabel("Acurácia")
    ax.margins(y=0.15)

    porcentagem(ax)
    preparar_eixo(ax)
    legenda(ax)
    salvar(fig, "fig2_acuracia_por_ano")


def fig_regime():
    dados = filtrar(
        pd.read_csv(config.RESULTADOS / "metricas_resumo.csv")
    )
    dados = dados.loc[dados["segmento"].eq("regime_vol")]

    if dados.empty:
        print("  Sem resultados por regime de volatilidade.")
        return

    tabela = dados.pivot(
        index="valor", columns="modelo", values="acuracia"
    ).reindex(["baixa", "alta"])

    modelos = modelos_presentes(tabela)
    posicoes = np.arange(len(tabela))
    deslocamentos = np.linspace(-0.16, 0.16, len(modelos))

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    for modelo, deslocamento in zip(modelos, deslocamentos):
        nome, cor, marcador, _ = ESTILOS[modelo]
        ax.scatter(
            posicoes + deslocamento,
            tabela[modelo],
            label=nome,
            color=cor,
            marker=marcador,
            s=35,
            zorder=3,
        )

    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=0.8)
    ax.set_xticks(
        posicoes, ["Volatilidade baixa", "Volatilidade alta"]
    )
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Acurácia")
    ax.margins(y=0.3)

    porcentagem(ax)
    preparar_eixo(ax)
    legenda(ax)
    salvar(fig, "fig3_acuracia_por_regime")


def fig_importancia():
    caminho = config.RESULTADOS / "importancia_variaveis.csv"

    if not caminho.exists():
        print("  Arquivo de importância das variáveis não encontrado.")
        return

    dados = pd.read_csv(caminho)
    dados = dados.loc[dados["conjunto"].eq(CONJUNTO)].copy()

    if dados.empty:
        print(f"  Sem importâncias para o conjunto {CONJUNTO}.")
        return

    dados = dados.sort_values("importancia_rf", ascending=False)
    rotulos = dados["variavel"].map(nome_variavel)
    tecnicos = set(config.VARIAVEIS_TECNICAS)

    cores = [
        "#275D80" if variavel in tecnicos else "#929292"
        for variavel in dados["variavel"]
    ]

    fig, ax = plt.subplots(
        figsize=(6.5, max(3.4, 0.25 * len(dados) + 0.8))
    )

    ax.barh(
        rotulos,
        dados["importancia_rf"],
        color=cores,
        height=0.6,
    )
    # referência: importância que cada variável teria se todas fossem iguais
    ax.axvline(1 / len(dados), color="#888888", linestyle=":", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Importância média por redução de impureza")
    ax.set_ylabel("")
    ax.set_xlim(left=0)

    preparar_eixo(ax, "x")

    if CONJUNTO == "ampliado":
        ax.legend(
            handles=[
                Patch(color="#275D80", label="Indicadores técnicos"),
                Patch(color="#929292", label="Variáveis de risco e retorno"),
            ],
            loc="lower right",
        )

    salvar(fig, "fig4_importancia_variaveis")


def ler_curvas(ativo):
    curvas = pd.read_parquet(
        config.RESULTADOS / "curvas_backtest.parquet"
    )
    curvas = curvas.loc[curvas["ativo"].eq(ativo)].copy()
    curvas["data"] = pd.to_datetime(curvas["data"])
    return curvas.sort_values("data").set_index("data")


def series_backtest(curvas):
    """Lista (coluna, nome, cor, linha) das estratégias presentes nas curvas."""
    series = [
        ("Comprar e manter", "Comprar e manter", COR_MANTER, "-"),
        ("CDI", "CDI", COR_CDI, "--"),
    ]
    for modelo, (nome, cor, _, linha) in ESTILOS.items():
        if modelo != "Sempre alta":
            series.append((f"{modelo} [{CONJUNTO}]", nome, cor, linha))
    return [s for s in series if s[0] in curvas.columns]


def marcar_reservado(ax, curvas, texto=True):
    inicio = pd.Timestamp(config.INICIO_RESERVADO)
    if not (curvas.index.min() <= inicio <= curvas.index.max()):
        return
    ax.axvline(inicio, color="#888888", linestyle=":", linewidth=0.8)
    if texto:
        ax.annotate(
            "Início do período reservado",
            xy=(inicio, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(-5, 0),
            textcoords="offset points",
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.8, "pad": 1.5},
        )


def fig_backtest(ativo):
    curvas = ler_curvas(ativo)

    if curvas.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    for coluna, nome, cor, linha in series_backtest(curvas):
        ax.plot(
            curvas.index,
            curvas[coluna],
            label=nome,
            color=cor,
            linestyle=linha,
            linewidth=1.1,
        )

    marcar_reservado(ax, curvas)

    ax.set_xlabel("Ano")
    ax.set_ylabel("Patrimônio acumulado (base 1)")
    ax.text(
        0.02, 0.96, ativo,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda valor, _: f"{valor:.1f}".replace(".", ","))
    )
    ax.margins(x=0.02)

    preparar_eixo(ax)
    legenda(ax)
    salvar(fig, f"fig5_backtest_{ativo}")


def fig_correlacao(ativo="BOVA11"):
    caminho = config.DADOS_PROCESSADOS / "dataset.parquet"

    if not caminho.exists():
        print("  Dataset não encontrado; matriz de correlação não gerada.")
        return

    dados = pd.read_parquet(caminho)
    dados["data"] = pd.to_datetime(dados["data"])

    selecao = (
        dados["ativo"].eq(ativo)
        & dados["data"].between(
            pd.Timestamp(config.DATA_INICIO),
            pd.Timestamp(config.DATA_FIM),
        )
    )
    dados = dados.loc[selecao]

    if dados.empty:
        print(f"  Sem observações para a correlação de {ativo}.")
        return

    correlacao = dados[config.VARIAVEIS_TECNICAS].corr(method="pearson")
    correlacao = correlacao.rename(
        index=nome_variavel, columns=nome_variavel
    )

    # A diagonal é mantida; a metade superior é ocultada.
    mascara = np.triu(
        np.ones(correlacao.shape, dtype=bool), k=1
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    sns.heatmap(
        correlacao,
        mask=mascara,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": "Correlação de Pearson",
            "shrink": 0.75,
            "ticks": [-1, -0.5, 0, 0.5, 1],
        },
        ax=ax,
    )

    ax.set_title(f"Correlação entre indicadores — {ativo}", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis="both", length=0)

    salvar(fig, f"fig6_correlacao_{ativo}")


# ---------------------------------------------------------------------------
# Figura 7: painel-resumo de um ativo (tearsheet)
# ---------------------------------------------------------------------------
def retornos_diarios(patrimonio):
    """Recupera o retorno de cada dia a partir da curva de patrimônio (base 1)."""
    anterior = patrimonio.shift(1).fillna(1.0)
    return patrimonio / anterior - 1


def acumular(retornos, regra):
    """Retorno composto por período (regra 'ME' = mês, 'YE' = ano)."""
    return (1 + retornos).resample(regra).prod() - 1


def fig_painel(ativo, modelo="Regressao Logistica"):
    """Painel com seis gráficos sobre um ativo e um modelo.

    Linha 1: patrimônio acumulado e perda a partir do pico (drawdown)
    Linha 2: retornos mensais (mapa de calor) e retornos anuais
    Linha 3: distribuição dos retornos mensais e acurácia móvel de 6 meses
    """
    curvas = ler_curvas(ativo)
    if curvas.empty or modelo not in ESTILOS:
        return

    nome_mod, cor_mod, _, _ = ESTILOS[modelo]
    col_mod = f"{modelo} [{CONJUNTO}]"
    if col_mod not in curvas.columns:
        return

    ret_mod = retornos_diarios(curvas[col_mod])
    ret_manter = retornos_diarios(curvas["Comprar e manter"])
    ret_cdi = retornos_diarios(curvas["CDI"])

    fig = plt.figure(figsize=(7.2, 9.6))
    grade = fig.add_gridspec(4, 2, height_ratios=[1.25, 0.65, 1.1, 1.0],
                             hspace=0.6, wspace=0.28, top=0.93)
    fig.suptitle(
        f"{ativo} — estratégia com {nome_mod} (custo de "
        f"{config.CUSTO_POR_TROCA:.2%} por troca)".replace(".", ","),
        x=0.06, ha="left", fontsize=11, fontweight="bold", y=0.995,
    )

    # 1) Patrimônio acumulado (largura total)
    ax = fig.add_subplot(grade[0, :])
    for coluna, nome, cor, linha in series_backtest(curvas):
        ax.plot(curvas.index, curvas[coluna], label=nome, color=cor,
                linestyle=linha, linewidth=1.0)
        ax.annotate(f"{curvas[coluna].iloc[-1]:.2f}".replace(".", ","),
                    xy=(curvas.index[-1], curvas[coluna].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=7, color=cor, va="center")
    marcar_reservado(ax, curvas, texto=False)
    titulo_painel(ax, "Patrimônio acumulado (base 1)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.margins(x=0.01)
    preparar_eixo(ax)
    ax.legend(loc="upper left", ncol=2)

    # 2) Perda a partir do pico (drawdown), largura total
    ax = fig.add_subplot(grade[1, :], sharex=fig.axes[0])
    for serie, nome, cor in ((curvas["Comprar e manter"], "Comprar e manter", COR_MANTER),
                             (curvas[col_mod], nome_mod, cor_mod)):
        queda = serie / serie.cummax() - 1
        ax.fill_between(queda.index, queda, 0, color=cor, alpha=0.15, linewidth=0)
        ax.plot(queda.index, queda, color=cor, linewidth=0.7, label=nome)
    titulo_painel(ax, "Perda a partir do pico (drawdown)")
    porcentagem(ax, casas=0)
    preparar_eixo(ax)
    ax.legend(loc="lower right", ncol=2)

    # 3) Mapa de calor dos retornos mensais da estratégia
    ax = fig.add_subplot(grade[2, 0])
    mensal = acumular(ret_mod, "ME")
    mapa = (mensal.to_frame("r")
            .assign(ano=mensal.index.year, mes=mensal.index.month)
            .pivot(index="ano", columns="mes", values="r")
            .reindex(columns=range(1, 13)))
    limite = np.nanmax(np.abs(mapa.values))
    sns.heatmap(
        mapa * 100, ax=ax, cmap="RdBu", center=0,
        vmin=-100 * limite, vmax=100 * limite,
        annot=True, fmt=".1f", annot_kws={"fontsize": 5.5},
        linewidths=0.4, linecolor="white", cbar=False,
    )
    ax.set_xticklabels(MESES, rotation=0, fontsize=6.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)
    titulo_painel(ax, "Retornos mensais da estratégia (%)")

    # 4) Retornos anuais: estratégia, comprar e manter e CDI
    ax = fig.add_subplot(grade[2, 1])
    anual = pd.DataFrame({
        nome_mod: acumular(ret_mod, "YE"),
        "Comprar e manter": acumular(ret_manter, "YE"),
        "CDI": acumular(ret_cdi, "YE"),
    })
    anual.index = anual.index.year
    posicoes = np.arange(len(anual))
    altura = 0.26
    for k, (coluna, cor) in enumerate(((nome_mod, cor_mod),
                                       ("Comprar e manter", COR_MANTER),
                                       ("CDI", COR_CDI))):
        ax.barh(posicoes + (k - 1) * altura, anual[coluna], height=altura,
                color=cor, label=coluna)
    ax.axvline(0, color="#555555", linewidth=0.6)
    ax.set_yticks(posicoes, anual.index)
    ax.invert_yaxis()
    porcentagem(ax, "x", casas=0)
    preparar_eixo(ax, "x")
    titulo_painel(ax, "Retornos anuais")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=6.5, handlelength=1.2, columnspacing=0.8)

    # 5) Distribuição dos retornos mensais
    ax = fig.add_subplot(grade[3, 0])
    mensal_manter = acumular(ret_manter, "ME")
    caixas = np.histogram_bin_edges(pd.concat([mensal, mensal_manter]), bins=20)
    ax.hist(mensal_manter, bins=caixas, color=COR_MANTER, alpha=0.35,
            label="Comprar e manter")
    ax.hist(mensal, bins=caixas, color=cor_mod, alpha=0.75, label=nome_mod)
    ax.axvline(mensal.mean(), color=cor_mod, linestyle="--", linewidth=0.8)
    porcentagem(ax, "x", casas=0)
    ax.set_ylabel("Número de meses")
    preparar_eixo(ax)
    titulo_painel(ax, "Distribuição dos retornos mensais")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              fontsize=6.5, handlelength=1.2)

    # 6) Acurácia móvel de 126 pregões (cerca de 6 meses)
    ax = fig.add_subplot(grade[3, 1])
    prev = pd.read_parquet(config.RESULTADOS / "previsoes.parquet")
    prev = prev.loc[prev["ativo"].eq(ativo)].copy()
    prev["data"] = pd.to_datetime(prev["data"])
    prev = prev.sort_values("data").set_index("data")
    janela = 126
    for chave in config.MODELOS:
        nome_res = MODELOS[chave][0]
        nome, cor, _, linha = ESTILOS[nome_res]
        acerto = (prev[f"pred_{CONJUNTO}__{chave}"] == prev["alvo"]).astype(float)
        ax.plot(acerto.rolling(janela).mean(), color=cor, linestyle=linha,
                linewidth=0.9, label=nome)
    base = (prev["pred_sempre_alta"] == prev["alvo"]).astype(float)
    ax.plot(base.rolling(janela).mean(), color="#717171", linestyle=":",
            linewidth=0.9, label="Sempre alta")
    ax.axhline(0.5, color="#888888", linewidth=0.6)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    porcentagem(ax, casas=0)
    preparar_eixo(ax)
    titulo_painel(ax, "Acurácia móvel (126 pregões)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=6.5, handlelength=1.6, columnspacing=0.8)

    salvar(fig, f"fig7_painel_{ativo}", ajustar=False)


def main():
    config.FIGURAS.mkdir(parents=True, exist_ok=True)

    fig_acuracia_ativo()
    fig_acuracia_ano()
    fig_regime()
    fig_importancia()

    ativos = pd.read_csv(
        config.RESULTADOS / "metricas_por_ativo.csv"
    )["ativo"].unique()

    selecionados = [
        ativo for ativo in ("BOVA11", "IVVB11")
        if ativo in ativos
    ]

    for ativo in selecionados or list(ativos[:1]):
        fig_backtest(ativo)
        fig_painel(ativo)

    fig_correlacao("BOVA11")

    print(f"\nFiguras salvas em: {config.FIGURAS}")


if __name__ == "__main__":
    main()
