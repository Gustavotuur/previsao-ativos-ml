# Guia: gerar as evidências e publicar o projeto

Cada passo abaixo gera uma **evidência para a monografia**. Tire os prints indicados e guarde-os em uma pasta `evidencias/` (fora do repositório).

## 1. Testes automatizados

```powershell
cd C:\TCC\tcc_previsao
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -v
```

📸 **Print 1:** o terminal mostrando `8 passed`.

## 2. Medição de desempenho

```powershell
python medir_desempenho.py
```

📸 **Print 2:** o resumo final, com os tempos, o ganho e a máquina. Isso também gera o `resultados/desempenho.csv`.

## 3. Versionamento com Git

1. Instale o Git: https://git-scm.com/download/win. Use as opções padrão.
2. Antes do primeiro commit, apague da pasta `LEIA-ME.md`, `00_escolher_fundos.py` e `04_figuras_versao_gustavo.py`.
3. Configure o Git uma única vez. Para não expor o seu e-mail, use o endereço "noreply" do GitHub
   (no GitHub: Settings → Emails → marque "Keep my email addresses private" e copie o endereço
   que termina em `@users.noreply.github.com`):
   ```powershell
   git config --global user.name "Gustavo Pereira da Silva"
   git config --global user.email "NUMERO+usuario@users.noreply.github.com"
   ```
4. Crie o repositório local e faça o primeiro commit:
   ```powershell
   cd C:\TCC\tcc_previsao
   git init
   git add .
   git commit -m "Versao inicial do pipeline, testes e aplicacao web"
   ```

📸 **Print 3:** a saída do `git log --oneline`.

> Daqui em diante, faça um commit a cada mudança relevante, por exemplo `git commit -am "Ajusta figura de backtest"`. O histórico de commits é a evidência de como o projeto evoluiu.

## 4. Publicação no GitHub (open source)

1. Crie uma conta em https://github.com, se ainda não tiver.
2. Clique em **New repository**:
   - nome: `previsao-ativos-ml`;
   - visibilidade: **Public**;
   - **não** marque a opção de criar README, porque o projeto já tem um.
3. Copie os comandos que o GitHub mostrar na seção "push an existing repository". São parecidos com:
   ```powershell
   git remote add origin https://github.com/SEU-USUARIO/previsao-ativos-ml.git
   git branch -M main
   git push -u origin main
   ```

📸 **Print 4:** a página do repositório no GitHub, mostrando os arquivos e o README.

## 5. Publicação da aplicação web (Streamlit Community Cloud)

1. Acesse https://share.streamlit.io e entre com a conta do GitHub.
2. Clique em **Create app**:
   - repositório: `previsao-ativos-ml`;
   - branch: `main`;
   - arquivo principal: `app.py`.
3. Clique em **Deploy** e aguarde a instalação. Leva alguns minutos.

📸 **Print 5:** a aplicação aberta no navegador, com o endereço público (`https://...streamlit.app`) visível.

Anote o endereço: ele vai para a monografia e para a apresentação.

## 6. Atualizar os dados publicados

```powershell
python atualizar.py --hoje
git add .
git commit -m "Atualizacao dos dados ate hoje"
git push
```

O Streamlit Community Cloud detecta o push e atualiza a aplicação sozinho.
