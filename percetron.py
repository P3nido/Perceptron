import random #Utilizei para gerar numeros aleatorios nos pesos iniciais e no tetha, para cada treinamento
import time #Utilizei para dar um delay em cada plotagem de grafico, pois como rodei um for 5 vezes ele estava dando problema ao exibir os graficos
import atexit #Utilizei para garantir que o arquivo txt que armazena a saida de prints no meu codigo seja fechado corretamente, apenas para log
import sys #Utilizei para copiar a saida dos prints do arquivo e redirecionar para um txt, para ter um log de todo o meu codigo
from pathlib import Path #Utilizei apenas para criar o caminho do meu arquivo de log saida_executcao.txt
import matplotlib.pyplot as plt #biblioteca para plotar os graficos 
import pandas as pd #biblioteca para organizar os resultados que preenchem a planilha
from openpyxl import load_workbook #biblioteca para escrever os resultados no arquivo xlsx mantendo o layout
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score #biblioteca para calculo das metricas da validacao


'''Classe e funcao para redirecionar a saída do print para um arquivo de log, mantendo a saída no console.'''

class Tee:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


arquivo_saida = Path(__file__).with_name("saida_execucao.txt")
_stdout_original = sys.stdout
_arquivo_log = open(arquivo_saida, "w", encoding="utf-8")
sys.stdout = Tee(_stdout_original, _arquivo_log)


def _finalizar_log_saida():
    sys.stdout = _stdout_original
    _arquivo_log.close()


atexit.register(_finalizar_log_saida)


def preencher_planilha_questao_2(caminho_planilha, resultados_df):
    if not caminho_planilha.exists():
        print(f"Planilha nao encontrada: {caminho_planilha}")
        return

    workbook = load_workbook(caminho_planilha)
    planilha = workbook["Sheet1"] if "Sheet1" in workbook.sheetnames else workbook.active
    linha_inicial = 11

    for idx, (_, resultado) in enumerate(resultados_df.iterrows()):
        linha_excel = linha_inicial + idx
        planilha.cell(row=linha_excel, column=7, value=f"{int(resultado['treinamento'])}° Treinamento")
        planilha.cell(row=linha_excel, column=9, value=float(resultado["w0_inicial"]))
        planilha.cell(row=linha_excel, column=10, value=float(resultado["w1_inicial"]))
        planilha.cell(row=linha_excel, column=11, value=float(resultado["w2_inicial"]))
        planilha.cell(row=linha_excel, column=12, value=float(resultado["w3_inicial"]))
        planilha.cell(row=linha_excel, column=13, value=float(resultado["w0_final"]))
        planilha.cell(row=linha_excel, column=14, value=float(resultado["w1_final"]))
        planilha.cell(row=linha_excel, column=15, value=float(resultado["w2_final"]))
        planilha.cell(row=linha_excel, column=16, value=float(resultado["w3_final"]))
        planilha.cell(row=linha_excel, column=17, value=int(resultado["numero_epocas"]))

    workbook.save(caminho_planilha)
    print(f"Planilha atualizada: {caminho_planilha.name}")


def preencher_planilha_questao_4(caminho_planilha, resultados_validacao_df):
    if not caminho_planilha.exists():
        print(f"Planilha nao encontrada: {caminho_planilha}")
        return

    workbook = load_workbook(caminho_planilha)
    planilha = workbook["Sheet1"] if "Sheet1" in workbook.sheetnames else workbook.active
    linha_inicial = 7
    coluna_inicial = 11  # K = y(t1)

    for linha in range(linha_inicial, linha_inicial + 10):
        for coluna in range(coluna_inicial, coluna_inicial + 5):
            planilha.cell(row=linha, column=coluna, value=None)

    for idx, (_, resultado) in enumerate(resultados_validacao_df.iterrows()):
        linha_excel = linha_inicial + idx
        for treino in range(1, 6):
            coluna_excel = coluna_inicial + (treino - 1)
            coluna_df = f"y_t{treino}"
            valor = resultado.get(coluna_df)
            if pd.notna(valor):
                planilha.cell(row=linha_excel, column=coluna_excel, value=float(valor))

    workbook.save(caminho_planilha)
    print(f"Planilha atualizada: {caminho_planilha.name}")

''' Fase de treinamento do Perceptron com os dados fornecidos'''
dados_treinamento = [
    [-0.6508, 0.1097, 4.0009, -1.0], [-1.4492, 0.8896, 4.4005, -1.0],
    [2.0850, 0.6876, 12.0710, -1.0], [0.2626, 1.1476, 7.7985, 1.0],
    [0.6418, 1.0234, 7.0427, 1.0], [0.2569, 0.6730, 8.3265, -1.0],
    [1.1155, 0.6043, 7.4446, 1.0], [0.0914, 0.3399, 7.0677, -1.0],
    [0.0121, 0.5256, 4.6316, 1.0], [-0.0429, 0.4660, 5.4323, 1.0],
    [0.4340, 0.6870, 8.2287, -1.0], [0.2735, 1.0287, 7.1934, 1.0],
    [0.4839, 0.4851, 7.4850, -1.0], [0.4089, -0.1267, 5.5019, -1.0],
    [1.4391, 0.1614, 8.5843, -1.0], [-0.9115, -0.1973, 2.1962, -1.0],
    [0.3654, 1.0475, 7.4858, 1.0], [0.2144, 0.7515, 7.1699, 1.0],
    [0.2013, 1.0014, 6.5489, 1.0], [0.6483, 0.2183, 5.8991, 1.0],
    [-0.1147, 0.2242, 7.2435, -1.0], [-0.7970, 0.8795, 3.8762, 1.0],
    [-1.0625, 0.6366, 2.4707, 1.0], [0.5307, 0.1285, 5.6883, 1.0],
    [-1.2200, 0.7777, 1.7252, 1.0], [0.3957, 0.1076, 5.6623, -1.0],
    [-0.1013, 0.5989, 7.1812, -1.0], [2.4482, 0.9455, 11.2095, 1.0],
    [2.0149, 0.6192, 10.9263, -1.0], [0.2012, 0.2611, 5.4631, 1.0]
]

coeficiente_de_aprendizado = 0.25 
caminho_planilha_q2 = Path(__file__).with_name("questao-2.xlsx")
caminho_planilha_q4 = Path(__file__).with_name("questao-4.xlsx")
pasta_graficos = Path(__file__).with_name("graficos")
pasta_graficos.mkdir(exist_ok=True)
resultados_treinamentos = []
resultados_validacao_df = pd.DataFrame({"amostra": list(range(1, 11))})
resultados_u_validacao_df = pd.DataFrame({"amostra": list(range(1, 11))})
resultados_metricas = []
resultados_validacao_por_treinamento = []

# Classe positiva de interesse: P2 (+1.0)
y_real_tabela_2 = [-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0]

tabela_2 = [
    [-0.3665, 0.0620, 5.9891], [-0.7842, 1.1267, 5.5912],
    [0.3012, 0.5611, 5.8234], [0.7757, 1.0648, 8.0677],
    [0.1570, 0.8028, 6.3040], [-0.7014, 1.0316, 3.6005],
    [0.3748, 0.1536, 6.1537], [-0.6920, 0.9404, 4.4058],
    [-1.3970, 0.7141, 4.9263], [-1.8842, -0.2805, 1.2548]
]

'''5 treinamentos (for de 0 a 4)'''
for sessao in range(0, 5):
    time.sleep(2)
    
    pesos = [random.uniform(-1, 1) for _ in range(3)] 
    tetha = random.uniform(-1, 1) 
    pesos_iniciais = pesos.copy()
    tetha_inicial = tetha
    
    print(f"Treinamento de numero: {sessao + 1}")
    print(f"Vetor de Pesos Iniciais: w0(theta)={tetha:.5f}, w1={pesos[0]:.5f}, w2={pesos[1]:.5f}, w3={pesos[2]:.5f}")
    
    n_epoca_sessao = 0
    historico_erros = []
    
    while True:
        '''variavel apenas para contar os erros da epoca para no final plotar a evolução no gráfico'''
        erros_vistos = 0 
        for linha in dados_treinamento:
            u_temp = sum(xi * wi for xi, wi in zip(linha[0:3], pesos)) + (tetha * -1)
            y_temp = 1.0 if u_temp >= 0 else -1.0
            if y_temp != linha[3]:
                erros_vistos += 1
        historico_erros.append(erros_vistos)

        erro_na_epoca = False
        for linha in dados_treinamento:
            x = linha[0:3]
            d = linha[3]
            
            u = sum(xi * wi for xi, wi in zip(x, pesos)) + (tetha * -1)
            y = 1.0 if u >= 0 else -1.0
            
            if y != d:
                erro_na_epoca = True
                for i in range(3):
                    pesos[i] = pesos[i] + coeficiente_de_aprendizado * (d - y) * x[i]
                tetha = tetha + coeficiente_de_aprendizado * (d - y) * (-1)
        
        n_epoca_sessao += 1
        
        if not erro_na_epoca:
            break

    print(f"Numero de epocas: {n_epoca_sessao}")
    print(f"Vetor de Pesos Finais: w0(theta)={tetha:.5f}, w1={pesos[0]:.5f}, w2={pesos[1]:.5f}, w3={pesos[2]:.5f}")
    print("-" * 50)

    resultado_atual = {
        "treinamento": sessao + 1,
        "w0_inicial": round(tetha_inicial, 5),
        "w1_inicial": round(pesos_iniciais[0], 5),
        "w2_inicial": round(pesos_iniciais[1], 5),
        "w3_inicial": round(pesos_iniciais[2], 5),
        "w0_final": round(tetha, 5),
        "w1_final": round(pesos[0], 5),
        "w2_final": round(pesos[1], 5),
        "w3_final": round(pesos[2], 5),
        "numero_epocas": n_epoca_sessao,
    }
    resultados_treinamentos.append(resultado_atual)
    resultados_treinamentos_df = pd.DataFrame(resultados_treinamentos)
    preencher_planilha_questao_2(caminho_planilha_q2, resultados_treinamentos_df)

    resultados_validacao_sessao = []
    u_validacao_sessao = []
    for x_teste in tabela_2:
        u_validacao = sum(xi * wi for xi, wi in zip(x_teste, pesos)) + (tetha * -1)
        y_validacao = 1.0 if u_validacao >= 0 else -1.0
        u_validacao_sessao.append(u_validacao)
        resultados_validacao_sessao.append(y_validacao)

    resultados_validacao_por_treinamento.append({
        "treinamento": sessao + 1,
        "y_pred": resultados_validacao_sessao.copy(),
        "u_pred": u_validacao_sessao.copy()
    })

    matriz_confusao = confusion_matrix(y_real_tabela_2, resultados_validacao_sessao, labels=[-1.0, 1.0])
    tn, fp, fn, tp = matriz_confusao.ravel()

    acertos = sum(1 for y_real, y_pred in zip(y_real_tabela_2, resultados_validacao_sessao) if y_real == y_pred)
    erros = len(y_real_tabela_2) - acertos

    acuracia = accuracy_score(y_real_tabela_2, resultados_validacao_sessao)
    sensibilidade = recall_score(y_real_tabela_2, resultados_validacao_sessao, pos_label=1.0, zero_division=0)
    especificidade = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precisao = precision_score(y_real_tabela_2, resultados_validacao_sessao, pos_label=1.0, zero_division=0)

    resultados_metricas.append({
        "treinamento": sessao + 1,
        "epocas": int(n_epoca_sessao),
        "w0_final": round(tetha, 5),
        "w1_final": round(pesos[0], 5),
        "w2_final": round(pesos[1], 5),
        "w3_final": round(pesos[2], 5),
        "acertos": int(acertos),
        "erros": int(erros),
        "Verdadeiro Negativo": int(tn),
        "Falso Positivo": int(fp),
        "Falso Negativo": int(fn),
        "Verdadeiro Positivo": int(tp),
        "acuracia": round(acuracia, 4),
        "sensibilidade": round(sensibilidade, 4),
        "especificidade": round(especificidade, 4),
        "precisao": round(precisao, 4),
    })

    print(f"Metricas de validacao - Treinamento {sessao + 1}")
    print("Matriz de Confusao (linhas = y_real [-1,+1], colunas = y_pred [-1,+1]):")
    print(matriz_confusao)
    print(f"Numero de Acertos: {acertos} | Numero de Erros: {erros}")
    print(
        f"Acuracia: {acuracia:.4f} | Sensibilidade: {sensibilidade:.4f} | "
        f"Especificidade: {especificidade:.4f} | Precisao: {precisao:.4f}"
    )
    print("-" * 50)

    resultados_validacao_df[f"y_t{sessao + 1}"] = resultados_validacao_sessao
    resultados_u_validacao_df[f"u_t{sessao + 1}"] = [round(valor, 4) for valor in u_validacao_sessao]
    preencher_planilha_questao_4(caminho_planilha_q4, resultados_validacao_df)


    ''' plotando a evolução de aprendizado para cada treinamento'''
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, n_epoca_sessao + 1), historico_erros, color='blue')
    plt.title(f'Evolucao do Erro - Treinamento {sessao + 1}')
    plt.xlabel('Epoca')
    plt.ylabel('Quantidade de Erros')
    plt.grid(True)

    caminho_grafico = pasta_graficos / f"grafico_treinamento_{sessao + 1}.png"
    plt.tight_layout()
    plt.savefig(caminho_grafico, dpi=150)
    print(f"Grafico salvo em: {caminho_grafico.name}")

    plt.close()

print("\nAPRENDIZADO FINALIZADO!")
time.sleep(2)


'''Fase de validação final do Perceptron usando os pesos do ultimo treinamento.'''

# tabela_2 = [
#     [-0.6508, 0.1097, 4.0009], [-1.4492, 0.8896, 4.4005],
#     [2.0850, 0.6876, 12.0710], [0.2626, 1.1476, 7.7985],
#     [0.6418, 1.0234, 7.0427], [0.2569, 0.6730, 8.3265],
#     [1.1155, 0.6043, 7.4446], [0.0914, 0.3399, 7.0677],
#     [0.0121, 0.5256, 4.6316], [-0.0429, 0.4660, 5.4323],
#     [0.4340, 0.6870, 8.2287], [0.2735, 1.0287, 7.1934],
#     [0.4839, 0.4851, 7.4850], [0.4089, -0.1267, 5.5019],
#     [1.4391, 0.1614, 8.5843], [-0.9115, -0.1973, 2.1962],
#     [0.3654, 1.0475, 7.4858], [0.2144, 0.7515, 7.1699],
#     [0.2013, 1.0014, 6.5489], [0.6483, 0.2183, 5.8991],
#     [-0.1147, 0.2242, 7.2435], [-0.7970, 0.8795, 3.8762],
#     [-1.0625, 0.6366, 2.4707], [0.5307, 0.1285, 5.6883],
#     [-1.2200, 0.7777, 1.7252], [0.3957, 0.1076, 5.6623],
#     [-0.1013, 0.5989, 7.1812], [2.4482, 0.9455, 11.2095],
#     [2.0149, 0.6192, 10.9263], [0.2012, 0.2611, 5.4631]
# ] 

print("-" * 40)
print("RESULTADOS DA VALIDACAO POR TREINAMENTO:")
for resultado_treino in resultados_validacao_por_treinamento:
    treino = resultado_treino["treinamento"]
    predicoes = resultado_treino["y_pred"]
    print(f"Treinamento {treino}:")
    for i, y_op in enumerate(predicoes, 1):
        classe = "P2" if y_op == 1 else "P1"
        print(f"Amostra {i}: y = {y_op} -> Classe {classe}")
    print("-" * 40)

print("\nRESUMO DAS METRICAS DAS 5 REDES (atividade 5):")
resumo_metricas_df = pd.DataFrame(resultados_metricas)
print(resumo_metricas_df.to_string(index=False))

print("\nTabela de saidas y por treinamento (y_t1...y_t5):")
print(resultados_validacao_df.to_string(index=False))

# Gera uma tabela visual com as metricas das 5 redes e salva na pasta graficos
fig, ax = plt.subplots(figsize=(16, 4.5))
ax.axis("off")

tabela_visual = ax.table(
    cellText=resumo_metricas_df.values,
    colLabels=resumo_metricas_df.columns,
    loc="center",
    cellLoc="center"
)
tabela_visual.auto_set_font_size(False)
tabela_visual.set_fontsize(9)
tabela_visual.scale(1.0, 1.4)

plt.title("Resumo das Metricas - 5 Treinamentos", fontsize=12, pad=12)
plt.tight_layout()

caminho_tabela_metricas = pasta_graficos / "tabela_metricas_treinamentos.png"
plt.savefig(caminho_tabela_metricas, dpi=200, bbox_inches="tight")
print(f"Tabela visual salva em: {caminho_tabela_metricas.name}")
plt.close()

# Gera tabela visual das classes previstas por amostra em cada treinamento
fig_y, ax_y = plt.subplots(figsize=(10, 4.5))
ax_y.axis("off")

tabela_y_visual = ax_y.table(
    cellText=resultados_validacao_df.values,
    colLabels=resultados_validacao_df.columns,
    loc="center",
    cellLoc="center"
)
tabela_y_visual.auto_set_font_size(False)
tabela_y_visual.set_fontsize(9)
tabela_y_visual.scale(1.0, 1.4)

plt.title("Saidas de Validacao por Treinamento", fontsize=12, pad=12)
plt.tight_layout()

caminho_tabela_y = pasta_graficos / "tabela_saidas_validacao.png"
plt.savefig(caminho_tabela_y, dpi=200, bbox_inches="tight")
print(f"Tabela visual salva em: {caminho_tabela_y.name}")
plt.close()

# Gera tabela visual dos valores de ativacao u por amostra e treinamento
fig_u, ax_u = plt.subplots(figsize=(10, 4.5))
ax_u.axis("off")

tabela_u_visual = ax_u.table(
    cellText=resultados_u_validacao_df.values,
    colLabels=resultados_u_validacao_df.columns,
    loc="center",
    cellLoc="center"
)
tabela_u_visual.auto_set_font_size(False)
tabela_u_visual.set_fontsize(9)
tabela_u_visual.scale(1.0, 1.4)

plt.title("Valores de Ativacao (u) por Treinamento", fontsize=12, pad=12)
plt.tight_layout()

caminho_tabela_u = pasta_graficos / "tabela_ativacao_u_validacao.png"
plt.savefig(caminho_tabela_u, dpi=200, bbox_inches="tight")
print(f"Tabela visual salva em: {caminho_tabela_u.name}")
plt.close()