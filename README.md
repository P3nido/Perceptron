# 🧠 Classificador Perceptron de Camada Única: Análise de Petróleo

Este repositório contém o código-fonte e a documentação do 1º Projeto Prático da disciplina de Redes Neurais (Engenharia da Computação - UFMG, 2026). 

O projeto consiste na implementação do zero (from scratch) do algoritmo **Perceptron de Camada Única**, utilizando a regra de aprendizado de Rosenblatt para classificar amostras de petróleo em duas categorias (P1 e P2) com base em seus atributos físico-químicos.

## 🎯 Objetivo do Projeto
Demonstrar o funcionamento prático e a matemática por trás das Redes Neurais Artificiais clássicas, sem depender de frameworks de alto nível (como TensorFlow ou PyTorch) para a etapa de treinamento. O algoritmo ajusta iterativamente os pesos sinápticos para encontrar um hiperplano capaz de separar linearmente os dados.

## ✨ Principais Funcionalidades Implementadas
- **Treinamento Manual:** Loop de aprendizado com ajuste de pesos baseado no erro $(d - y)$.
- **Múltiplas Sessões:** Execução de 5 treinamentos independentes, com inicialização aleatória de pesos no intervalo `[-1, 1]`.
- **Semente Aleatória:** Implementação de um *delay* de tempo entre os treinamentos para garantir a dispersão real das sementes da biblioteca `random`.
- **Trava de Segurança (Early Stopping):** O algoritmo para ao atingir erro zero ou ao bater o teto de segurança de 1000 épocas (evitando loops infinitos).
- **Cálculo de RMSE:** Acompanhamento matemático da descida do erro através da Raiz do Erro Quadrático Médio em cada época.
- **Automação de Relatórios:** Uso do `pandas` e `openpyxl` para preencher automaticamente planilhas `.xlsx` com os pesos iniciais, pesos finais e validações, sem quebrar a formatação do Excel.
- **Geração de Gráficos:** Plotagem e salvamento automático das curvas de aprendizado (Erro Absoluto e RMSE) e Matrizes de Confusão com o `matplotlib`.
- **Logs de Execução:** Espelhamento da saída do console para um arquivo `saida_execucao.txt`.

## 📈 Resultados Obtidos
O modelo demonstrou que os dados fornecidos são perfeitamente separáveis de forma linear. A rede neural atingiu a convergência em todos os 5 treinamentos (geralmente entre 380 e 440 épocas).

Na fase de validação, testada contra dados empíricos não vistos no treinamento, o Perceptron obteve **100% de aproveitamento**, apresentando pontuação máxima (1.0) nas seguintes métricas do `scikit-learn`:
- Acurácia
- Sensibilidade (Recall)
- Especificidade
- Precisão

## 🚀 Como executar o projeto

### Pré-requisitos
Certifique-se de ter o Python instalado na sua máquina e instale as bibliotecas necessárias. Recomenda-se o uso de um ambiente virtual (venv).

```bash
pip install pandas openpyxl matplotlib scikit-learn
