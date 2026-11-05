"""
config.py - CONFIGURACAO CENTRAL DO TCC
======================================

Previsao de Movimentos de Ativos Financeiros com Machine Learning
Aluno: Gustavo Pereira da Silva - Orientadora: Profa. Ana Maria dos Santos Vianna

Reune as decisoes do experimento (amostra, datas, variaveis, modelos e validacao).
Os demais scripts importam este modulo, de modo que a metodologia descrita na
monografia e o codigo usam os mesmos parametros.
"""
import os
from datetime import date
from pathlib import Path

# ===========================================================================
# PASTAS (relativas a pasta do projeto)
# ===========================================================================
RAIZ = Path(__file__).resolve().parent
DADOS_BRUTOS = RAIZ / "dados" / "brutos"            # dados baixados
DADOS_PROCESSADOS = RAIZ / "dados" / "processados"  # dados limpos + indicadores
RESULTADOS = RAIZ / "resultados"                    # metricas e previsoes
FIGURAS = RAIZ / "figuras"                          # graficos da monografia

# ===========================================================================
# PERIODO
# ===========================================================================
# O download comeca um ano antes da amostra para o aquecimento dos indicadores:
# alguns so tem valor depois de acumular historico (o TRIX encadeia tres medias
# de 20 dias). Em 01/01/2018 todos ja estao definidos.
DATA_INICIO_DOWNLOAD = "2017-01-01"
DATA_INICIO = "2018-01-01"            # inicio da amostra analisada

# Data final fixa do experimento, para que os resultados da monografia possam ser
# reproduzidos em qualquer data.
DATA_FIM_EXPERIMENTO = "2026-09-30"

# Modo plataforma ("python atualizar.py --hoje" define TCC_ATE_HOJE): usa os dados
# ate a data atual. Esses resultados nao fazem parte da avaliacao da monografia.
DATA_FIM = date.today().isoformat() if os.environ.get("TCC_ATE_HOJE") else DATA_FIM_EXPERIMENTO

# Periodo final reservado (o ultimo ano). E previsto normalmente, sempre com treino
# apenas no passado, mas nenhuma decisao do trabalho foi tomada com base nele;
# e reportado separadamente.
INICIO_RESERVADO = "2025-10-01"

# ===========================================================================
# AMOSTRA
# ===========================================================================
# Ticker na B3 -> descricao. Todos tem abertura, maxima, minima e fechamento diarios.
# So entram no estudo os que passam nos filtros de historico e liquidez (abaixo).
ETFS = {
    "BOVA11": "iShares Ibovespa",
    "SMAL11": "iShares Small Cap",
    "IVVB11": "iShares S&P 500 em reais",
    "DIVO11": "It Now IDIV (dividendos)",
}

ACOES = {
    "PETR4": "Petrobras PN (petroleo)",
    "VALE3": "Vale ON (mineracao)",
    "ITUB4": "Itau Unibanco PN (bancos)",
    "WEGE3": "WEG ON (bens de capital)",
}

# Fundos de investimento imobiliario negociados em bolsa
FIIS = {
    "HGLG11": "CSHG Logistica",
    "KNRI11": "Kinea Renda Imobiliaria",
}

# Series auxiliares: baixadas, mas nao previstas pelos modelos.
INDICE_VIX = "^VIX"       # volatilidade implicita do S&P 500 (usa-se a variacao diaria)
BENCHMARK = "^BVSP"       # Ibovespa, apenas para comparacao em graficos

# Filtros de entrada na amostra. Observacao: sao avaliados no periodo inteiro, o que
# introduz vies de sobrevivencia (discutido nas limitacoes da monografia).
TOLERANCIA_DIAS = 30            # dados ate 30 dias depois do inicio e ate 30 dias antes do fim
LIQUIDEZ_MINIMA = 1_000_000     # volume financeiro diario mediano minimo, em R$

# ===========================================================================
# DOWNLOAD
# ===========================================================================
MAX_WORKERS = 4          # downloads simultaneos (ThreadPoolExecutor)
TENTATIVAS = 3           # tentativas por serie em caso de falha de rede

# ===========================================================================
# PROBLEMA DE CLASSIFICACAO
# ===========================================================================
# alvo = 1 (alta) se o retorno de t ate t+HORIZONTE for positivo; 0 caso contrario.
HORIZONTE = 1            # proximo pregao (D+1)

# Variaveis de entrada: variacao do VIX e nove indicadores tecnicos
# (colunas calculadas em 02_preparar_variaveis.py).
VARIAVEIS_TECNICAS = ["vix_var", "cci", "macd", "williams_r", "stoch_smi", "trix",
                    "bollinger_pctb", "cmo", "dpo", "roc"]

# Metricas de risco e retorno (extensao opcional, desligada).
VARIAVEIS_RISCO_RETORNO = ["ret_5d", "ret_21d", "ret_63d", "ret_126d",
                           "vol_21d", "vol_63d", "vol_252d",
                           "dist_mm50", "dist_mm200", "drawdown_252d", "rsi_14"]

CONJUNTOS = {
    "tecnicos": VARIAVEIS_TECNICAS,
    # "ampliado": VARIAVEIS_TECNICAS + VARIAVEIS_RISCO_RETORNO,
}
TODAS_VARIAVEIS = list(dict.fromkeys(v for vs in CONJUNTOS.values() for v in vs))

# Hiperparametros fixados antes dos testes, sem busca automatica, para reduzir o
# risco de ajustar os modelos ao periodo de teste.
MODELOS = ["logistica", "random_forest"]
# Extensao opcional: "rede_neural" (MLP)

# ===========================================================================
# VALIDACAO WALK-FORWARD (janela expansiva)
# ===========================================================================
# Treino inicial de TREINO_INICIAL pregoes e teste nos JANELA_TESTE seguintes; a
# janela testada entra no treino e o processo se repete ate o fim da amostra. Cada
# janela de teste e prevista por um modelo treinado apenas com pregoes anteriores a ela.
TREINO_INICIAL = 504     # ~2 anos de pregoes
JANELA_TESTE = 126       # ~6 meses

# Regime de volatilidade (calculado em 02): alta quando a volatilidade de 63 dias do
# ativo supera a mediana do seu proprio historico ate o dia anterior.

SEMENTE = 42             # semente fixa (amostragem da Random Forest)

CUSTO_POR_TROCA = 0.001  # 0,10% por troca de posicao no backtest (ativo <-> CDI)

# Sensibilidade aos custos por troca. 0,03% corresponde as tarifas da B3 para pessoa
# fisica (negociacao + liquidacao); os valores maiores incluem spread e corretagem.
CUSTOS_SENSIBILIDADE = [0.0, 0.0003, 0.0005, 0.001, 0.002]

# Estrategias aleatorias de referencia: decidem ao acaso, a cada dia, entre o ativo e
# o CDI, com a mesma proporcao de dias no ativo que o modelo. Separam o efeito da
# previsao do efeito da exposicao ao CDI.
N_ESTRATEGIAS_ALEATORIAS = 500

# Bootstrap em blocos: os testes binomial e de McNemar supoem previsoes independentes,
# mas os ativos de um mesmo pregao se movem juntos e dias proximos sao dependentes.
# Sorteiam-se blocos de pregoes consecutivos (com todos os ativos de cada pregao).
BLOCO_BOOTSTRAP = 21       # pregoes por bloco (~1 mes)
N_BOOTSTRAP = 2000         # reamostragens
