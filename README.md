# Previsão de Movimentos de Ativos Financeiros com Machine Learning

Trabalho de Conclusão de Curso, Bacharelado em Ciência da Computação, Universidade Veiga de Almeida (UVA), 2026.
Autor: **Gustavo Pereira da Silva**. Orientadora: **Profa. MSc Ana Maria dos Santos Vianna**.

Este sistema reprodutível prevê a direção do próximo pregão (alta ou baixa) de 10 ativos da B3 com **Regressão Logística** e **Random Forest**. Ele cobre todas as etapas:
- coleta concorrente de dados públicos;
- dez variáveis explicativas (nove indicadores técnicos e a variação do VIX), calculadas com medidas para reduzir o vazamento temporal, com verificação automática;
- avaliação por **validação walk-forward**;
- testes estatísticos (binomial e McNemar);
- uma **aplicação web em Streamlit**.

> Material educacional e de pesquisa. **Não é recomendação de investimento.**
>
> Os preços foram obtidos do Yahoo Finance e a taxa do CDI do Banco Central do Brasil. Os dados processados incluídos neste repositório servem exclusivamente para reproduzir os resultados de um trabalho acadêmico.

## Arquitetura

```
Yahoo Finance / Banco Central (SGS 12)
        │  01_baixar_dados.py      coleta concorrente (ThreadPoolExecutor)
        ▼
dados/brutos/*.csv
        │  02_preparar_variaveis.py   limpeza, filtros, 10 variáveis, alvo
        ▼
dados/processados/dataset.parquet
        │  03_modelar.py           walk-forward, modelos, métricas, testes (joblib)
        ▼
resultados/*.csv|parquet ──► 04_figuras.py (figuras)   ──► app.py (Streamlit)
```

Todos os parâmetros (amostra, datas, variáveis e modelos) ficam em `config.py`. O `atualizar.py` roda as etapas em sequência. O arquivo `dados/processados/dataset.parquet`, incluído no repositório, é o conjunto de dados congelado usado na monografia: `python atualizar.py` parte dele e reproduz exatamente os resultados. Uma nova coleta pode trazer valores levemente diferentes, porque o Yahoo Finance reajusta a série histórica a cada novo dividendo.

## Como executar

Requisitos: Python 3.11 ou mais novo (os resultados foram gerados com Python 3.12.10).

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python atualizar.py            # reproduz os resultados da monografia (dados congelados)
python atualizar.py --preparar # recalcula as variaveis a partir dos dados brutos ja baixados
python atualizar.py --baixar   # baixa os dados de novo (novo conjunto; resultados podem mudar)
python atualizar.py --hoje     # modo plataforma: dados e previsoes ate hoje
streamlit run app.py           # abre a aplicação web
python -m pytest -v            # testes automatizados
python medir_desempenho.py     # mede o ganho do paralelismo
```

O notebook `estudo_passo_a_passo.ipynb` percorre o pipeline de forma didática.

## Experimento e plataforma

- **Resultados do experimento:** gerados com a base congelada de 01/01/2018 a 30/09/2026 (`DATA_FIM_EXPERIMENTO` em `config.py`). São os números da monografia e os que a aplicação mostra por padrão.
- **Atualização da plataforma:** `python atualizar.py --hoje` baixa os dados até a data atual e gera novas previsões. Esses resultados não fazem parte da avaliação da monografia.

A aplicação indica qual dos dois casos está sendo exibido e, separadamente, a data do último download dos dados.

## Estrutura do repositório

| Caminho | Conteúdo |
|---|---|
| `config.py` | Parâmetros do experimento |
| `01_baixar_dados.py` | Coleta concorrente de cotações e do CDI |
| `02_preparar_variaveis.py` | Limpeza, filtros, indicadores técnicos e alvo |
| `03_modelar.py` | Walk-forward, Regressão Logística, Random Forest, métricas, testes e backtest |
| `04_figuras.py` | Figuras em 300 DPI |
| `app.py` | Aplicação web (Streamlit) |
| `atualizar.py` | Orquestra as etapas |
| `medir_desempenho.py` | Benchmark sequencial × paralelo |
| `tests/` | Testes automatizados (pytest) |
| `dados/processados/`, `resultados/`, `figuras/` | Saídas usadas pela aplicação publicada |

## Decisões de método

- **Problema:** classificação binária. O alvo é 1 se o fechamento do próximo pregão for maior que o de hoje.
- **Medidas para reduzir o vazamento temporal:** os indicadores são calculados apenas com dados até o fechamento do dia da previsão, o VIX entra com 1 dia de defasagem e o DPO é calculado na versão causal. O teste `test_indicadores_nao_usam_o_futuro` recalcula os indicadores com a série cortada em várias datas e confere se os valores não mudam. Essas verificações cobrem os indicadores e o walk-forward; a seleção da amostra, porém, usou critérios de histórico e liquidez avaliados no período inteiro (viés de sobrevivência, discutido na monografia).
- **Validação walk-forward** com janela expansiva: 504 pregões de treino inicial e testes de 126. O período a partir de 01/10/2025 é reservado e reportado à parte.
- **Hiperparâmetros fixados antes dos testes**, sem busca, e semente aleatória fixa.
- **Referências de comparação:** "sempre alta", classe majoritária e persistência.

## Versões utilizadas

Versões instaladas no ambiente em que os resultados da monografia foram gerados (Windows, Python 3.12.10):

| Biblioteca | Versão |
|---|---|
| pandas | 3.1.0 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| joblib | 1.6.0 |
| yfinance | 1.7.0 |
| curl_cffi | 0.17.0 |
| requests | 2.34.2 |
| certifi | 2026.10.23 |
| pyarrow | 26.0.0 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| streamlit | 1.66.1 |
| pytest | 9.2.0 |

O `requirements.txt` fixa a versão do Streamlit (a mesma usada na aplicação publicada) e indica versões mínimas para as demais.

## Licença

[MIT](LICENSE)
