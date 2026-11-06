"""
Orquestra as etapas do sistema: coleta, preparacao, modelagem e figuras.

Uso:
  python atualizar.py              experimento da monografia: reproduz os resultados a partir
                                   do conjunto de dados congelado (dados/processados)
  python atualizar.py --preparar   experimento, recalculando as variaveis a partir dos dados
                                   brutos ja baixados (dados/brutos)
  python atualizar.py --baixar     experimento, baixando os dados de novo (gera um NOVO
                                   conjunto congelado; os resultados podem mudar levemente)
  python atualizar.py --hoje       plataforma: baixa os dados ate hoje e gera as previsoes do
                                   proximo pregao (sempre baixa)

Por que o experimento usa um conjunto de dados congelado: o Yahoo Finance fornece precos
AJUSTADOS por dividendos e desdobramentos. Quando um ativo paga um novo dividendo, toda a
serie passada e reajustada, e uma nova coleta com a mesma data final pode trazer valores
levemente diferentes. O arquivo dados/processados/dataset.parquet, publicado no repositorio,
e o conjunto usado na monografia; a partir dele, as etapas 03 e 04 reproduzem exatamente os
resultados.

Para atualizacao diaria automatica no Windows, agende  python atualizar.py --hoje  no
Agendador de Tarefas (ex.: dias uteis as 19h, depois do fechamento da B3).
"""
# Executa as etapas em subprocessos, em sequencia; uma falha interrompe a execucao.
import os
import subprocess
import sys
import time
from pathlib import Path

PASSOS = ["01_baixar_dados.py", "02_preparar_variaveis.py", "03_modelar.py", "04_figuras.py"]


def escolher_passos(pasta):
    """Decide quais etapas rodar conforme o modo pedido e os dados disponiveis."""
    tem_brutos = any((pasta / "dados" / "brutos" / "precos").glob("*.csv"))
    tem_processados = (pasta / "dados" / "processados" / "dataset.parquet").exists()
    if "--hoje" in sys.argv or "--baixar" in sys.argv:
        return PASSOS, "coleta nova"
    if "--preparar" in sys.argv:
        if not tem_brutos:
            sys.exit("Nao ha dados brutos em dados/brutos/precos. Use --baixar.")
        return PASSOS[1:], "variaveis recalculadas a partir dos dados brutos"
    if tem_processados:
        return PASSOS[2:], "conjunto de dados congelado (dados/processados)"
    if tem_brutos:
        return PASSOS[1:], "dados brutos ja baixados"
    return PASSOS, "primeira execucao: coleta completa"


def main():
    pasta = Path(__file__).resolve().parent
    inicio = time.time()
    env = dict(os.environ)
    if "--hoje" in sys.argv:
        env["TCC_ATE_HOJE"] = "1"
    passos, modo = escolher_passos(pasta)
    print(f"Modo: {modo}.")
    for passo in PASSOS:
        if passo not in passos:
            print(f"\n===== {passo}: pulado =====")
            continue
        print(f"\n===== {passo} =====")
        r = subprocess.run([sys.executable, str(pasta / passo)], cwd=pasta, env=env)
        if r.returncode != 0:
            sys.exit(f"Falha em {passo}. Corrija o erro acima e rode de novo.")
    print(f"\nPlataforma atualizada em {(time.time() - inicio) / 60:.1f} min. "
          "Abra com:  streamlit run app.py")


if __name__ == "__main__":
    main()
