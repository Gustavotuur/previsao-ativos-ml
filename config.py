"""
config.py - CONFIGURACAO CENTRAL DO TCC
======================================

Previsao de Movimentos de Ativos Financeiros com Machine Learning
Aluno: Gustavo Pereira da Silva - Orientadora: Profa. Ana Maria dos Santos Vianna

PARA QUE SERVE ESTE ARQUIVO
---------------------------
Ele nao executa nada: so guarda DECISOES. Todos os outros scripts fazem "import config"
e leem daqui a amostra, as datas, as variaveis e os parametros dos modelos.
Vantagem: a metodologia descrita na monografia e o codigo dizem exatamente a mesma
coisa, e para mudar o experimento (por exemplo, incluir um ativo) basta mexer aqui.
"""
import os                     # acesso a variaveis de ambiente do sistema operacional
from datetime import date     # para saber a data de hoje
from pathlib import Path      # forma moderna de montar caminhos de pastas e arquivos

# ===========================================================================
# PASTAS
# ===========================================================================
# __file__ e o caminho deste proprio arquivo (config.py). .resolve() transforma em
# caminho absoluto e .parent pega a pasta onde ele esta, ou seja, a pasta do projeto.
# Assim os scripts funcionam em qualquer computador, sem caminho fixo tipo "C:\TCC".
RAIZ = Path(__file__).resolve().parent

# O operador "/" do Path junta pedacos de caminho: RAIZ / "dados" -> C:\TCC\tcc_previsao\dados
DADOS_BRUTOS = RAIZ / "dados" / "brutos"            # o que foi baixado da internet
DADOS_PROCESSADOS = RAIZ / "dados" / "processados"  # dados limpos + indicadores
RESULTADOS = RAIZ / "resultados"                    # tabelas de metricas e previsoes
FIGURAS = RAIZ / "figuras"                          # graficos para a monografia

# ===========================================================================
# PERIODO
# ===========================================================================
# Baixamos um ano a mais (2017) do que analisamos (2018). Motivo: alguns indicadores
# so passam a existir depois de acumular historico (ex.: o TRIX usa tres medias de 20
# dias encadeadas). Esse ano extra e o "aquecimento": em 01/01/2018 tudo ja tem valor.
DATA_INICIO_DOWNLOAD = "2017-01-01"
DATA_INICIO = "2018-01-01"            # inicio da amostra analisada

# Data final FIXA do experimento: assim, quem rodar o codigo daqui a um ano obtem os
# mesmos numeros que estao na monografia (reprodutibilidade).
DATA_FIM_EXPERIMENTO = "2026-09-30"

# Modo plataforma: "python atualizar.py --hoje" cria a variavel de ambiente TCC_ATE_HOJE.
# Se ela existir, usamos os dados ate hoje (para a previsao do dia seguinte no site);
# se nao existir, usamos a data fixa do experimento.
# Sintaxe: "valor_se_verdadeiro if condicao else valor_se_falso"
DATA_FIM = date.today().isoformat() if os.environ.get("TCC_ATE_HOJE") else DATA_FIM_EXPERIMENTO

# Periodo final RESERVADO (o ultimo ano). Os modelos o preveem normalmente, sempre
# treinando so com o passado, mas nenhuma decisao do trabalho e tomada olhando para
# ele. Funciona como uma "prova final" e e reportado separadamente na monografia.
INICIO_RESERVADO = "2025-10-01"

# ===========================================================================
# AMOSTRA
# ===========================================================================
# Dicionarios Python: {chave: valor}. A chave e o codigo (ticker) na B3 e o valor e
# uma descricao. Todos tem abertura, maxima, minima e fechamento diarios, entao todos
# os indicadores tecnicos podem ser calculados para todos.
# So entram no estudo os que passarem nos filtros de historico e liquidez (mais abaixo).
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

# Fundos de investimento imobiliario: sao fundos de investimento, mas
# negociados em bolsa como acoes, por isso tem os mesmos dados diarios.
FIIS = {
    "HGLG11": "CSHG Logistica",
    "KNRI11": "Kinea Renda Imobiliaria",
}
# Para ampliar a amostra depois, basta acrescentar tickers acima (ex.: BOVV11, PIBB11,
# BRAX11, FIND11, MATB11, ISUS11; BBAS3, ABEV3; XPML11, MXRF11).

# Series auxiliares: baixadas, mas NAO sao previstas pelos modelos.
INDICE_VIX = "^VIX"       # "indice do medo" americano (usamos a variacao diaria)
BENCHMARK = "^BVSP"       # Ibovespa, so para comparacao em graficos

# Filtros de entrada na amostra (criterios objetivos, descritos na metodologia)
TOLERANCIA_DIAS = 30            # o ativo precisa ter dados ate 30 dias depois do inicio
                                # e ate 30 dias antes do fim (historico completo)
LIQUIDEZ_MINIMA = 1_000_000     # volume financeiro diario mediano minimo, em R$
                                # (o "_" e so separador visual: 1_000_000 = 1000000)

# ===========================================================================
# DOWNLOAD
# ===========================================================================
MAX_WORKERS = 4          # quantos downloads acontecem ao mesmo tempo (concurrent.futures)
TENTATIVAS = 3           # quantas vezes tentar de novo se a internet falhar

# ===========================================================================
# PROBLEMA DE CLASSIFICACAO
# ===========================================================================
# O modelo responde a pergunta: "o preco daqui a HORIZONTE pregoes sera maior que hoje?"
# alvo = 1 (alta) se o retorno de t ate t+HORIZONTE for positivo; 0 (baixa) caso contrario.
# HORIZONTE = 1 significa prever o proximo pregao (D+1).
HORIZONTE = 1

# As "pistas" que o modelo recebe: 10 indicadores classicos de analise tecnica.
# Cada nome corresponde a uma coluna calculada em 02_preparar_variaveis.py.
VARIAVEIS_TECNICAS = ["vix_var", "cci", "macd", "williams_r", "stoch_smi", "trix",
                    "bollinger_pctb", "cmo", "dpo", "roc"]

# Metricas extras de risco e retorno (extensao opcional, desligada por padrao).
VARIAVEIS_RISCO_RETORNO = ["ret_5d", "ret_21d", "ret_63d", "ret_126d",
                           "vol_21d", "vol_63d", "vol_252d",
                           "dist_mm50", "dist_mm200", "drawdown_252d", "rsi_14"]

# Conjuntos de variaveis a testar. Hoje so o de indicadores tecnicos esta ativo.
CONJUNTOS = {
    "tecnicos": VARIAVEIS_TECNICAS,
    # Extensao opcional: descomente para testar se risco e retorno acrescentam algo
    # "ampliado": VARIAVEIS_TECNICAS + VARIAVEIS_RISCO_RETORNO,
}
# Lista com todas as variaveis usadas em algum conjunto, sem repeticao.
# dict.fromkeys(...) remove duplicatas mantendo a ordem original.
TODAS_VARIAVEIS = list(dict.fromkeys(v for vs in CONJUNTOS.values() for v in vs))

# Modelos comparados. Os hiperparametros (configuracoes internas de cada modelo) sao
# fixados ANTES dos testes, sem busca automatica: assim nao ha como o modelo ser
# "ajustado ao teste", o que inflaria os resultados.
MODELOS = ["logistica", "random_forest"]
# Extensao opcional: acrescente "rede_neural" para incluir uma rede neural (MLP)

# ===========================================================================
# VALIDACAO WALK-FORWARD
# ===========================================================================
# Imagine um aluno que estuda o passado e faz provas sobre o futuro:
#   - estuda os primeiros TREINO_INICIAL pregoes (~2 anos) e faz a prova dos
#     JANELA_TESTE pregoes seguintes (~6 meses);
#   - depois esses 6 meses entram no material de estudo e ele faz a prova dos
#     proximos 6, e assim por diante ate o fim da amostra.
# O modelo NUNCA ve o futuro do dia que esta prevendo.
TREINO_INICIAL = 504     # ~2 anos de pregoes (cerca de 252 pregoes por ano)
JANELA_TESTE = 126       # ~6 meses

# Regime de volatilidade (calculado em 02): um dia e de volatilidade "alta" quando a
# volatilidade dos ultimos 63 dias do ativo esta acima da mediana do seu proprio
# passado ate o dia anterior. Serve para ver se os modelos funcionam melhor em
# mercado calmo ou agitado.

SEMENTE = 42             # semente aleatoria fixa: a Random Forest sorteia amostras;
                         # fixando a semente, os resultados saem iguais a cada execucao

CUSTO_POR_TROCA = 0.001  # 0,10% de custo a cada vez que o backtest troca de posicao
                         # (sai do ativo para o CDI ou vice-versa)

# Sensibilidade aos custos: o backtest e refeito com cada um destes custos por troca.
# 0,03% corresponde as tarifas da B3 para pessoa fisica (negociacao + liquidacao);
# os valores maiores incluem spread e corretagem.
CUSTOS_SENSIBILIDADE = [0.0, 0.0003, 0.0005, 0.001, 0.002]

# Estrategias aleatorias de referencia: decidem ao acaso, a cada dia, se ficam no
# ativo ou no CDI, com a MESMA proporcao de dias no ativo que o modelo. Servem para
# saber se o resultado do backtest vem da previsao ou so da exposicao ao CDI.
N_ESTRATEGIAS_ALEATORIAS = 500
