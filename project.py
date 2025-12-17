import kagglehub
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


path = kagglehub.dataset_download("valakhorasani/gym-members-exercise-dataset")

print("Path to dataset files:", path)

gym_data = pd.read_csv('C:/Users/35191/.cache/kagglehub/datasets/valakhorasani/gym-members-exercise-dataset/versions/1/gym_members_exercise_tracking.csv')

# print(gym_data.shape) #dimensões 973 linhas, 15 colunas

def hist():
    x=1
def bxpl():
    x=1

def barr():
    x=1
def pizza():
    x=1

def varNum():
    print("-------------------------------------------")
    print("Qual é o tipo de gráfico?")
    print("1 - Histograma")
    print("2 - Boxplot")
    opcao3 = float(input("Opção: "))
    while opcao3 != 1 and opcao3 != 2:
        print("ERROR: Valor deve estar entre 1 e 2")
        opcao3 = float(input("Opção: "))

    if opcao3 == 1:
        hist()
    else:
        bxpl()


def varCat():
    print("-------------------------------------------")
    print("Qual é o tipo de gráfico?")
    print("1 - Gráfico de Barras")
    print("2 - Gráfico Pizza")
    opcao3 = float(input("Opção: "))
    while opcao3 != 1 and opcao3 != 2:
        print("ERROR: Valor deve estar entre 1 e 2")
        opcao3 = float(input("Opção: "))    
    
    if opcao3 == 1:
        barr()
    else:
        pizza()


def viz():
    print("-------------------------------------------")
    print("Que variável pretende vizualizar?")
    print("1 - Idade")
    print("2 - Género")
    print("3 - Peso (kg)")
    print("4 - Altura")
    print("5 - BPM Máximo")
    print("6 - BPM Médio")
    print("7 - BPM de Descanso")
    print("8 - Duração da Sessão")
    print("9 - Calorias Queimadas")
    print("10 - Tipo de Treino")
    print("11 - Percentagem de Gordura Corporal")
    print("12 - Água ingerida")
    print("13 - Frequência de Treino")
    print("14 - Nível de Experiência")
    print("15 - Índice de Massa Corporal")
    opcao2 = float(input("Opção: "))
    while opcao2 < 1 or opcao2 > 15:
        print("ERROR: Valor deve estar entre 1 e 15")
        opcao2 = float(input("Opção: "))

    num = [1,3,4,5,6,7,8,9,11,12,13,15]

    if opcao2 in num:
        varNum()
    else:
        varCat()


def scatter():
    x=1
    print(x)

def barras():
    x=2
    print(x)

def boxplot():
    x=3
    print(x)


def comp():
    print("-------------------------------------------")
    print("Que variável pretende comparar?")
    print("Primeira variável")
    print("1 - Idade")
    print("2 - Género")
    print("3 - Peso (kg)")
    print("4 - Altura")
    print("5 - BPM Máximo")
    print("6 - BPM Médio")
    print("7 - BPM de Descanso")
    print("8 - Duração da Sessão")
    print("9 - Calorias Queimadas")
    print("10 - Tipo de Treino")
    print("11 - Percentagem de Gordura Corporal")
    print("12 - Água ingerida")
    print("13 - Frequência de Treino")
    print("14 - Nível de Experiência")
    print("15 - Índice de Massa Corporal")
    opcao2 = float(input("Opção: "))
    while opcao2 < 1 or opcao2 > 15:
        print("ERROR: Valor deve estar entre 1 e 15")
        opcao2 = float(input("Opção: "))

    print("-------------------------------------------")
    print("Segunda variável")
    print("1 - Idade")
    print("2 - Género")
    print("3 - Peso (kg)")
    print("4 - Altura")
    print("5 - BPM Máximo")
    print("6 - BPM Médio")
    print("7 - BPM de Descanso")
    print("8 - Duração da Sessão")
    print("9 - Calorias Queimadas")
    print("10 - Tipo de Treino")
    print("11 - Percentagem de Gordura Corporal")
    print("12 - Água ingerida")
    print("13 - Frequência de Treino")
    print("14 - Nível de Experiência")
    print("15 - Índice de Massa Corporal")
    opcao3 = float(input("Opção: "))
    while opcao3 < 1 or opcao3 > 15 or opcao3==opcao2:
        print("ERROR: Valor deve estar entre 1 e 15 e não pode ser igual à primeira variável")
        opcao3 = float(input("Opção: "))
    
    num = [1,3,4,5,6,7,8,9,11,12,13,15]

    if opcao2 in num and opcao3 in num:
        scatter()
    elif opcao2 not in num and opcao3 not in num:
        barras()
    else:
        boxplot()




    


def main():
    print("MENU:")
    print("O que pretende fazer?")
    print("1 - Vizualizar a distribuição de uma variável")
    print("2 - Comparar duas variáveis")
    opcao1 = float(input("Opção: "))
    while opcao1 != 1 and opcao1 != 2:
        print("ERROR: Valor deve estar entre 1 e 2")
        opcao1 = float(input("Opção: "))
    
    if opcao1 == 1:
        viz()
    else:
        comp()


if __name__ == "__main__":
    main()



