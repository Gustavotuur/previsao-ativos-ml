"""
Atualizacao completa da plataforma: baixa os dados, recalcula as variaveis, refaz a
validacao e as previsoes do proximo pregao, e gera as figuras.

Uso:
  python atualizar.py            experimento da monografia (dados ate DATA_FIM_EXPERIMENTO)
  python atualizar.py --baixar   experimento, baixando os dados de novo
  python atualizar.py --hoje     plataforma: dados e previsoes ate hoje (sempre baixa)

Por que o experimento NAO baixa os dados de novo por padrao: o Yahoo Finance fornece
precos AJUSTADOS por dividendos e desdobramentos. Quando um ativo paga um novo dividendo,
toda a serie passada e reajustada, e os resultados mudam um pouco, mesmo com a data final
fixa. Para que o experimento seja reprodutivel, os dados baixados uma vez ficam
congelados em dados/brutos e sao reutilizados.
Para atualizacao diaria automatica no Windows, agende  python atualizar.py --hoje  no
Agendador de Tarefas (ex.: dias uteis as 19h, depois do fechamento da B3).
"""
# Este arquivo so chama os outros scripts em sequencia, como se voce digitasse
# "python 01_...", "python 02_..." etc. no terminal. Se um falhar, para tudo.
import os
import subprocess
import sys
import time
from pathlib import Path

PASSOS = ["01_baixar_dados.py", "02_preparar_variaveis.py", "03_modelar.py", "04_figuras.py"]


def main():
    pasta = Path(__file__).resolve().parent
    inicio = time.time()
    env = dict(os.environ)
    if "--hoje" in sys.argv:
        env["TCC_ATE_HOJE"] = "1"
    ja_baixado = (pasta / "dados" / "brutos" / "relatorio_download.csv").exists()
    for passo in PASSOS:
        if (passo.startswith("01_") and ja_baixado and "--hoje" not in sys.argv
                and "--baixar" not in sys.argv):
            print(f"\n===== {passo}: pulado (dados congelados; use --baixar para baixar de novo) =====")
            continue
        print(f"\n===== {passo} =====")
        # sys.executable = o Python que esta rodando agora (o do .venv)
        r = subprocess.run([sys.executable, str(pasta / passo)], cwd=pasta, env=env)
        if r.returncode != 0:
            sys.exit(f"Falha em {passo}. Corrija o erro acima e rode de novo.")
    print(f"\nPlataforma atualizada em {(time.time() - inicio) / 60:.1f} min. "
          "Abra com:  streamlit run app.py")


if __name__ == "__main__":
    main()
