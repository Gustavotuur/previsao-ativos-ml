# Previsão de Movimentos de Ativos Financeiros com Machine Learning

Trabalho de Conclusão de Curso, Bacharelado em Ciência da Computação, Universidade Veiga de Almeida (UVA), 2026.
Autor: **Gustavo Pereira da Silva**. Orientadora: **Profa. Ana Maria dos Santos Vianna**.

Este sistema reprodutível prevê a direção do próximo pregão (alta ou baixa) de 10 ativos da B3 com **Regressão Logística** e **Random Forest**. Ele cobre todas as etapas:
- coleta concorrente de dados públicos;
- indicadores técnicos sem vazamento de dados;
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
        │  02_preparar_variaveis.py   limpeza, filtros, 10 indicadores, alvo
        ▼
dados/processados/dataset.parquet
        │  03_modelar.py           walk-forward, modelos, métricas, testes (joblib)
        ▼
resultados/*.csv|parquet ──► 04_figuras.py (figuras)   ──► app.py (Streamlit)
```

Todos os parâmetros (amostra, datas, variáveis e modelos) ficam em `config.py`. O `atualizar.py` roda as etapas em sequência. No modo experimento, os dados já baixados são reutilizados: o Yahoo Finance reajusta a série histórica a cada novo dividendo, e baixar de novo mudaria levemente os resultados.

## Como executar

Requisitos: Python 3.11 ou mais novo.

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python atualizar.py            # experimento completo (dados fixos até 30/09/2026)
python atualizar.py --baixar   # experimento, baixando os dados de novo
python atualizar.py --hoje     # modo plataforma: dados e previsões até hoje
streamlit run app.py           # abre a aplicação web
python -m pytest -v            # testes automatizados
python medir_desempenho.py     # mede o ganho do paralelismo
```

O notebook `estudo_passo_a_passo.ipynb` percorre o pipeline de forma didática.

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
- **Sem vazamento de dados:** os indicadores usam só dados até o dia da previsão. O VIX entra com 1 dia de defasagem e o DPO é calculado na versão causal. O teste `test_indicadores_nao_usam_o_futuro` verifica isso automaticamente.
- **Validação walk-forward** com janela expansiva: 504 pregões de treino inicial e testes de 126. O período a partir de 01/10/2025 é reservado e reportado à parte.
- **Hiperparâmetros fixados antes dos testes**, sem busca, e semente aleatória fixa.
- **Referências de comparação:** "sempre alta", classe majoritária e persistência.

## Licença

[MIT](LICENSE)
