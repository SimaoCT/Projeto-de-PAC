'''
Dataset Columns:
Person ID: An identifier for each individual.
Gender: The gender of the person (Male/Female).
Age: The age of the person in years.
Occupation: The occupation or profession of the person.
Sleep Duration (hours): The number of hours the person sleeps per day.
Quality of Sleep (scale: 1-10): A subjective rating of the quality of sleep, ranging from 1 to 10.
Physical Activity Level (minutes/day): The number of minutes the person engages in physical activity daily.
Stress Level (scale: 1-10): A subjective rating of the stress level experienced by the person, ranging from 1 to 10.
BMI Category: The BMI category of the person (e.g., Underweight, Normal, Overweight).
Blood Pressure (systolic/diastolic): The blood pressure measurement of the person, indicated as systolic pressure over diastolic pressure.
Heart Rate (bpm): The resting heart rate of the person in beats per minute.
Daily Steps: The number of steps the person takes per day.
Sleep Disorder: The presence or absence of a sleep disorder in the person (None, Insomnia, Sleep Apnea).

'''


#Library imports:

import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
import seaborn as sns
from tabulate import tabulate


#Variável Global
df = None

#Funções a Implementar

#Carregar dados através de ficheiro (obrigatório )
def carregar_dados():
    try:
        global df
        df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv", encoding= "utf-8")

        #Definir o índice como o Person ID
        df = df.set_index("Person ID")

        # Separar a coluna "Blood Pressure" em duas novas colunas numéricas
        df[["BP_Sys","BP_Dias"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)

        #Contar quais são nulos e substituir os nulos por None
        print(df.isnull().sum())
        df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")


    except FileNotFoundError:
        print("Verifica se o ficheiro está na mesma pasta ou se o diretório é o correto")


#Não está a funcionar
def imprimir_dados():
    if df == None:
        print("Importe os dados primeiro")
    else:
        print(tabulate(df.head(10), headers= "keys", tablefmt = "psql"))
        

#Menu para a parte de visualização de dados
def visualizacao():
    while True:
        print("\n1 - Tipos de dados e missing values")
        print("2 - Estatística descritiva")
        print("0 - Voltar ao menu")
        
        try:
            opt = int(input("Qual a sua opção? "))
        except ValueError:
            print("Tem de ser um número inteiro entre 0 e 4")
        
        if opt == 0:
            break
        elif opt == 1:
            print(df.info())
            print("Valores nulos:\n",df.isnull().sum())
        elif opt == 2:
            print(df.describe())
        else:
            print("Valor não reconhecido")

#Guardar em ficheiro (obrigatorio)      
def guardar():
    if df is None:
        print("Não existem dados para guardar.")
        return

    try:
        #Nome e tipo de ficheiro
        tipo_ficheiro = input("Tipo de formato (ex: csv, txt...): ").lower()
        nome_ficheiro = input(f"Nome do ficheiro (ex: dados.{tipo_ficheiro}): ")

        if tipo_ficheiro == "csv":
            df.to_csv(nome_ficheiro, index=False)
        
        elif tipo_ficheiro == "excel" or tipo_ficheiro == "xlsx":
            #é preciso instalar -> openpyxl
            #não está a funcionar na mesma
            df.to_excel(nome_ficheiro, index=False)
        
        else:
            print("Tipo de ficheiro não reconhecido.")
    
    except Exception as a:
        print(f"Erro ao guardar: {a}")

    



#Menu Global de Visualização (obrigatorio)
def main():
    while True:
        print("\n1 - Entrada dos dados")
        print("2 - Imprime dados")
        print("3 - Guardar dados em ficheiro")
        print("4 - Visualização")
        print("0 - Sair")

        try:
            opt = int(input("Qual a sua opção? "))
        except ValueError:
            print("Tem de ser um número inteiro entre 0 e 4")

        if opt == 0:
            break
        elif opt == 1:
            #Lê os dados a partir do ficheiro 
            carregar_dados()
            print("\nOs dados foram carregados com sucesso.")
        elif opt == 2:
            #print("Ainda não funciona")
            imprimir_dados()
        elif opt == 3:
            guardar()
        elif opt == 4:
            visualizacao()
        else:
            print("Valor não reconhecido")

main()
