"""
planilha_tendencia.py - PLANILHA DO TESTE DE R2 (chamada pelo analise_tendencia.py)
==================================================================================

Gera um Excel em que os ajustes e os R2 sao FORMULAS (INCLINACAO, INTERCEPCAO, RQUAD,
LN, EXP...), para que o resultado possa ser conferido sem Python. Os precos e as
probabilidades previstas sao os dados de entrada (em azul).
"""
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

FONTE = "Arial"
AZUL = Font(name=FONTE, color="0000FF")
NORMAL = Font(name=FONTE)
NEGRITO = Font(name=FONTE, bold=True)
TITULO = Font(name=FONTE, bold=True, size=13)
CABECALHO = PatternFill("solid", fgColor="DCE6F1")
DESTAQUE = PatternFill("solid", fgColor="FFF2CC")


def _cab(ws, linha, textos, col=1):
    for i, t in enumerate(textos):
        c = ws.cell(linha, col + i, t)
        c.font, c.fill = NEGRITO, CABECALHO
        c.alignment = Alignment(wrap_text=True, vertical="center")


def _larguras(ws, larguras):
    for i, w in enumerate(larguras, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _grafico(ws, n, ativo):
    ch = ScatterChart()
    ch.title = f"{ativo}: preço e ajustes linear e exponencial"
    ch.x_axis.title, ch.y_axis.title = "Tempo (anos desde o início)", "Preço ajustado (R$)"
    ch.style, ch.height, ch.width = 13, 9, 18
    x = Reference(ws, min_col=2, min_row=2, max_row=n + 1)
    for col, nome, cor, larg in [(3, "Preço", "A6A6A6", 9000), (5, "Linear", "1F4E79", 28000),
                                 (6, "Exponencial", "C55A11", 28000)]:
        s = Series(Reference(ws, min_col=col, min_row=2, max_row=n + 1), x, title=nome)
        s.marker.symbol = "none"
        s.graphicalProperties.line.solidFill = cor
        s.graphicalProperties.line.width = larg
        s.smooth = False
        ch.series.append(s)
    ch.roundedCorners = False
    ch.legend.position = "b"
    ch.legend.overlay = False
    ch.title.overlay = False
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ch.x_axis.number_format = "0"
    ch.y_axis.number_format = "0"
    ch.x_axis.scaling.min = 0
    ws.add_chart(ch, "H16")


def _aba_ativo(wb, ativo, g):
    ws = wb.create_sheet(ativo)
    g = g.sort_values("data").reset_index(drop=True)
    n = len(g)
    _cab(ws, 1, ["Data", "t (anos)", "Preço ajustado", "ln(Preço)", "Linear ajustado", "Exponencial ajustado"])
    ult = n + 1
    for i, (d, p) in enumerate(zip(g["data"], g["preco"]), start=2):
        ws.cell(i, 1, d.to_pydatetime()).number_format = "dd/mm/yyyy"
        ws.cell(i, 2, f"=(A{i}-$A$2)/365.25").number_format = "0.000"
        c = ws.cell(i, 3, float(p)); c.font = AZUL; c.number_format = "0.00"
        ws.cell(i, 4, f"=LN(C{i})").number_format = "0.0000"
        ws.cell(i, 5, f"=$J$3+$J$4*B{i}").number_format = "0.00"
        ws.cell(i, 6, f"=EXP($J$6+$J$7*B{i})").number_format = "0.00"
    T, P, L, E, LIN = (f"B2:B{ult}", f"C2:C{ult}", f"D2:D{ult}", f"F2:F{ult}", f"E2:E{ult}")
    ws["H1"] = "Parâmetros e R² (fórmulas)"; ws["H1"].font = TITULO
    linhas = [
        (3, "Linear: intercepto a", f"=INTERCEPT({P},{T})", "0.0000"),
        (4, "Linear: inclinação b (R$ por ano)", f"=SLOPE({P},{T})", "0.0000"),
        (6, "Exponencial: ln(A)", f"=INTERCEPT({L},{T})", "0.0000"),
        (7, "Exponencial: B (taxa contínua por ano)", f"=SLOPE({L},{T})", "0.0000"),
        (8, "Exponencial: crescimento anual = e^B − 1", "=EXP(J7)-1", "0.0%"),
        (10, "R² linear (escala do preço)", f"=RSQ({P},{T})", "0.000"),
        (11, "R² linear, conferência: 1 − SQres/SQtot", f"=1-SUMXMY2({P},{LIN})/DEVSQ({P})", "0.000"),
        (12, "R² exponencial (escala do preço)", f"=1-SUMXMY2({P},{E})/DEVSQ({P})", "0.000"),
        (13, "R² exponencial (escala do ln; é o da linha de tendência do Excel)", f"=RSQ({L},{T})", "0.000"),
    ]
    for r, rot, f, fmt in linhas:
        ws.cell(r, 8, rot).font = NORMAL
        c = ws.cell(r, 10, f); c.number_format = fmt; c.font = NORMAL
    for r in (10, 12):
        ws.cell(r, 8).font = NEGRITO; ws.cell(r, 10).font = NEGRITO; ws.cell(r, 10).fill = DESTAQUE
    ws["H14"] = ("Para comparar os modelos, use as linhas 10 e 12: as duas medem o erro na escala do preço. "
                 "A linha 13 mede o erro do logaritmo e não é comparável com a 10.")
    ws["H14"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("H14:N15")
    ws.row_dimensions[14].height = 30
    _larguras(ws, [12, 9, 13, 11, 14, 17, 2, 62, 2, 12])
    ws.freeze_panes = "A2"
    _grafico(ws, n, ativo)


def _aba_leia_me(wb):
    ws = wb.active
    ws.title = "Leia-me"
    textos = [
        ("Teste de R²: tendência linear x exponencial e R² dos classificadores", TITULO),
        ("TCC: Previsão de movimentos de ativos financeiros com machine learning — Gustavo Pereira da Silva (UVA, 2026)", NORMAL),
        ("Gerado pelo script analise_tendencia.py a partir da base congelada da monografia (01/01/2018 a 30/09/2026).", NORMAL),
        ("", NORMAL),
        ("1. Tendência de preço (uma aba por ativo)", NEGRITO),
        ("Linear: P = a + b·t. Exponencial: P = A·e^(B·t), ajustada como reta em ln(P): ln P = ln A + B·t. t = anos desde o primeiro pregão.", NORMAL),
        ("Os dois ajustes usam mínimos quadrados (INCLINAÇÃO e INTERCEPÇÃO). R² = 1 − SQ residual / SQ total.", NORMAL),
        ("Para comparar linear e exponencial, os dois R² são calculados na escala do preço. O R² da exponencial na escala do ln", NORMAL),
        ("(que é o que a linha de tendência exponencial do Excel mostra) mede outra coisa e não é comparável (Gujarati; Porter, 2009, p. 202-203).", NORMAL),
        ("A aba Resumo traz os R² com preços diários (fórmulas, das abas dos ativos), mensais e anuais (calculados pelo script).", NORMAL),
        ("", NORMAL),
        ("2. Passeio aleatório", NEGRITO),
        ("Séries simuladas sem tendência e sem previsibilidade, com a volatilidade do BOVA11. Mesmo assim a reta tem R² alto com frequência:", NORMAL),
        ("R² alto de preço x tempo não indica previsibilidade (regressão espúria; Gujarati; Porter, 2009, seção 21.7).", NORMAL),
        ("", NORMAL),
        ("3. RL x RF (classificadores)", NEGRITO),
        ("O alvo é 0/1, então o R² comum não se aplica (Gujarati; Porter, 2009, p. 546). Usam-se, fora da amostra, o Brier skill e o pseudo-R² de McFadden,", NORMAL),
        ("contra um modelo nulo que prevê sempre a proporção de altas do treino de cada janela. Valores perto de zero = sem ganho sobre o nulo.", NORMAL),
        ("", NORMAL),
        ("Convenções: números em azul são dados de entrada; números em preto são fórmulas.", NORMAL),
    ]
    for i, (t, f) in enumerate(textos, 1):
        c = ws.cell(i, 1, t); c.font = f
    ws.column_dimensions["A"].width = 150


def _aba_resumo(wb, tend, ativos):
    ws = wb.create_sheet("Resumo", 1)
    ws["A1"] = "R² da tendência: linear x exponencial"; ws["A1"].font = TITULO
    ws["A2"] = ("Diária: fórmulas que puxam as abas dos ativos. Mensal (último preço do mês) e anual (último preço do ano): "
                "calculados pelo analise_tendencia.py, mesmo método.")
    ws["A2"].font = NORMAL
    cab = ["Ativo", "R² linear (diária)", "R² exponencial (diária)", "Melhor (diária)",
           "R² linear (mensal)", "R² exponencial (mensal)", "R² linear (anual)", "R² exponencial (anual)",
           "Crescimento anual (exponencial)", "R² exponencial na escala do ln (diária)"]
    _cab(ws, 4, cab)
    ws.row_dimensions[4].height = 45
    for i, a in enumerate(ativos, start=5):
        ws.cell(i, 1, a).font = NEGRITO
        ws.cell(i, 2, f"='{a}'!$J$10").number_format = "0.000"
        ws.cell(i, 3, f"='{a}'!$J$12").number_format = "0.000"
        ws.cell(i, 4, f'=IF(ABS(B{i}-C{i})<0.01,"empate",IF(B{i}>C{i},"linear","exponencial"))')
        for col, freq, campo in [(5, "mensal", "r2_linear"), (6, "mensal", "r2_exp_escala_preco"),
                                 (7, "anual", "r2_linear"), (8, "anual", "r2_exp_escala_preco")]:
            v = tend[(tend["ativo"] == a) & (tend["frequencia"] == freq)][campo].iloc[0]
            c = ws.cell(i, col, float(v)); c.number_format = "0.000"; c.font = AZUL
        ws.cell(i, 9, f"='{a}'!$J$8").number_format = "0.0%"
        ws.cell(i, 10, f"='{a}'!$J$13").number_format = "0.000"
    m = 5 + len(ativos)
    ws.cell(m, 1, "Média").font = NEGRITO
    for col in (2, 3, 5, 6, 7, 8, 10):
        L = get_column_letter(col)
        c = ws.cell(m, col, f"=AVERAGE({L}5:{L}{m - 1})"); c.number_format = "0.000"; c.font = NEGRITO
    ws.cell(m + 1, 1, "Empate = diferença menor que 0,01. A média entre ativos é só um resumo: a comparação deve ser lida ativo a ativo.").font = NORMAL
    n_dias = tend[tend["frequencia"] == "diaria"]["n"]
    ws.cell(m + 2, 1, f"Pontos por ativo: diária ≈ {int(n_dias.min())}–{int(n_dias.max())}; mensal ≈ "
                      f"{int(tend[tend['frequencia'] == 'mensal']['n'].min())}; anual = "
                      f"{int(tend[tend['frequencia'] == 'anual']['n'].min())} (2018 a 2026, este último parcial).").font = NORMAL
    _larguras(ws, [12, 12, 14, 13, 12, 14, 12, 14, 14, 16])
    ws.freeze_panes = "B5"


def _aba_passeio(wb, pa, datas, exemplo):
    ws = wb.create_sheet("Passeio aleatório")
    ws["E1"] = "Controle: reta ajustada a um passeio aleatório"; ws["E1"].font = TITULO
    _cab(ws, 1, ["Data", "t (anos)", "Preço simulado", "Reta ajustada"])
    n = len(exemplo)
    for i, (d, p) in enumerate(zip(pd.to_datetime(datas), exemplo), start=2):
        ws.cell(i, 1, d.to_pydatetime()).number_format = "dd/mm/yyyy"
        ws.cell(i, 2, f"=(A{i}-$A$2)/365.25").number_format = "0.000"
        c = ws.cell(i, 3, float(p)); c.font = AZUL; c.number_format = "0.00"
        ws.cell(i, 4, f"=INTERCEPT($C$2:$C${n + 1},$B$2:$B${n + 1})+SLOPE($C$2:$C${n + 1},$B$2:$B${n + 1})*B{i}").number_format = "0.00"
    r = pa.iloc[0]
    itens = [
        ("R² da reta nesta série simulada (fórmula)", f"=RSQ(C2:C{n + 1},B2:B{n + 1})", "0.000", False),
        ("Volatilidade diária usada (desvio do log-retorno do BOVA11)", float(r["volatilidade_diaria"]), "0.00%", True),
        ("Séries simuladas pelo script", int(r["simulacoes"]), "0", True),
        ("R² mediano nas simulações", float(r["r2_mediano"]), "0.000", True),
        ("Proporção de simulações com R² > 0,5", float(r["proporcao_r2_acima_0_5"]), "0.0%", True),
    ]
    for k, (rot, v, fmt, entrada) in enumerate(itens, start=3):
        ws.cell(k, 5, rot).font = NORMAL
        c = ws.cell(k, 7, v); c.number_format = fmt; c.font = AZUL if entrada else NEGRITO
    ws["E9"] = ("Cada retorno diário é sorteado com média zero: não há tendência nem nada a prever. "
                "Mesmo assim a reta tem R² alto com frequência — por isso R² de preço x tempo não mede previsibilidade.")
    ws["E9"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("E9:K11")
    _larguras(ws, [12, 9, 14, 13, 50, 2, 10])
    ch = ScatterChart()
    ch.title = "Passeio aleatório simulado e reta ajustada"
    ch.style, ch.height, ch.width = 13, 8, 16
    x = Reference(ws, min_col=2, min_row=2, max_row=n + 1)
    for col, nome, cor in [(3, "Preço simulado", "A6A6A6"), (4, "Reta", "1F4E79")]:
        s = Series(Reference(ws, min_col=col, min_row=2, max_row=n + 1), x, title=nome)
        s.marker.symbol = "none"; s.graphicalProperties.line.solidFill = cor; s.smooth = False
        ch.series.append(s)
    ch.roundedCorners = False; ch.legend.position = "b"; ch.legend.overlay = False; ch.title.overlay = False
    ch.x_axis.delete = False; ch.y_axis.delete = False; ch.x_axis.title = "Tempo (anos)"
    ch.x_axis.number_format = "0"; ch.x_axis.scaling.min = 0
    ws.add_chart(ch, "E13")


def _aba_classificadores(wb, prev):
    ws = wb.create_sheet("RL x RF")
    p = prev.sort_values(["ativo", "data"]).reset_index(drop=True)
    n = len(p)
    _cab(ws, 1, ["Data", "Ativo", "Alvo (1 = alta)", "Prob. RL", "Prob. RF", "Prob. nulo (altas no treino)",
                 "Log-ver. RL", "Log-ver. RF", "Log-ver. nulo", "Acerto RL", "Acerto RF"])
    ws.row_dimensions[1].height = 45
    for i, row in enumerate(p.itertuples(index=False), start=2):
        ws.cell(i, 1, pd.Timestamp(row.data).to_pydatetime()).number_format = "dd/mm/yyyy"
        ws.cell(i, 2, row.ativo)
        for col, v in [(3, int(row.alvo)), (4, float(row.prob_tecnicos__logistica)),
                       (5, float(row.prob_tecnicos__random_forest)), (6, float(row.prob_nulo))]:
            c = ws.cell(i, col, v); c.font = AZUL
            if col > 3:
                c.number_format = "0.0000"
        for col, pc in [(7, "D"), (8, "E"), (9, "F")]:
            ws.cell(i, col, f"=C{i}*LN({pc}{i})+(1-C{i})*LN(1-{pc}{i})").number_format = "0.0000"
        ws.cell(i, 10, f"=IF((D{i}>=0.5)=(C{i}=1),1,0)")
        ws.cell(i, 11, f"=IF((E{i}>=0.5)=(C{i}=1),1,0)")
    u = n + 1
    ws["M1"] = "Resultado fora da amostra (fórmulas)"; ws["M1"].font = TITULO
    _cab(ws, 3, ["Medida", "Regressão Logística", "Random Forest", "Nulo"], col=13)
    linhas = [
        ("Acurácia", f"=AVERAGE(J2:J{u})", f"=AVERAGE(K2:K{u})", None, "0.00%"),
        ("Brier (erro quadrático da probabilidade)", f"=SUMXMY2(D2:D{u},C2:C{u})/COUNT(C2:C{u})",
         f"=SUMXMY2(E2:E{u},C2:C{u})/COUNT(C2:C{u})", f"=SUMXMY2(F2:F{u},C2:C{u})/COUNT(C2:C{u})", "0.00000"),
        ("Brier skill = 1 − Brier / Brier do nulo", "=1-N5/P5", "=1-O5/P5", "=1-P5/P5", "0.0000"),
        ("Log-verossimilhança", f"=SUM(G2:G{u})", f"=SUM(H2:H{u})", f"=SUM(I2:I{u})", "0.0"),
        ("Pseudo-R² de McFadden = 1 − LL / LL do nulo", "=1-N7/P7", "=1-O7/P7", "=1-P7/P7", "0.0000"),
    ]
    for k, (rot, a, b, c, fmt) in enumerate(linhas, start=4):
        ws.cell(k, 13, rot).font = NORMAL
        for col, f in [(14, a), (15, b), (16, c)]:
            if f:
                cel = ws.cell(k, col, f); cel.number_format = fmt
    for k in (6, 8):
        for col in (13, 14, 15):
            ws.cell(k, col).font = NEGRITO
            ws.cell(k, col).fill = DESTAQUE
    ws["M10"] = ("Valores negativos: fora da amostra, as probabilidades dos modelos foram um pouco piores do que usar "
                 "a proporção de altas do treino. O R² comum não se aplica a um alvo 0/1.")
    ws["M10"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("M10:P13")
    _larguras(ws, [12, 9, 9, 9, 9, 12, 10, 10, 10, 8, 8, 2, 40, 14, 14, 12])
    ws.freeze_panes = "A2"


def exportar(dados, tend, pa, datas_pa, exemplo_pa, clas, prev, caminho):
    wb = Workbook()
    _aba_leia_me(wb)
    ativos = sorted(dados["ativo"].unique())
    _aba_resumo(wb, tend, ativos)
    for a in ativos:
        _aba_ativo(wb, a, dados[dados["ativo"] == a])
    _aba_passeio(wb, pa, datas_pa, exemplo_pa)
    _aba_classificadores(wb, prev)
    wb.save(caminho)
