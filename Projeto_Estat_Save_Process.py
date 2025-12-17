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
import os

#Pré processamento

df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv", encoding= "utf-8")
#Definir o índice como o Person ID
df = df.set_index("Person ID")
# Separar a coluna "Blood Pressure" em duas novas colunas numéricas e remover a antiga
df[["BP_Sys","BP_Dias"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
df = df.drop(columns=["Blood Pressure"])
#Substituir os nulos por None
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
#Alterar na Variável BMI o Normal weight para Normal apenas
df["BMI Category"] = df["BMI Category"].replace("Normal Weight", "Normal")
# Vamos iterar sobre todas as colunas de variáveis categóricas para verificar as categorias presentes
for coluna in df.columns:
    #O .dtype verifica se é do tipo object (é como o pandas trata as colunas), e o nunique é para verificar se tem menos de 10 categorias (para incluir sleep quality e stress)
    if df[coluna].dtype == "object" or df[coluna].nunique() < 11:
        print(f"Frequências Absolutas de {coluna}\n")
        print(df[coluna].value_counts())
        print("-" * 35 + "\n")
#As variáveis categóricas têm de ser definidas antes da visualização gráfica para que por exemplo apareça o "Underweight" no BMI

#Guardar em ficheiro (obrigatorio)      

def guardar(df_save):
    tipo_ficheiro = input("Tipo de formato (ex: csv, txt...): ").lower().strip()
    nome_ficheiro = input(f"Nome do ficheiro (sem extensão): ").strip()


    #Verifica o tipo de ficheiro
    if tipo_ficheiro == "csv":
        nome_final = f"{nome_ficheiro}.csv"
    elif tipo_ficheiro == "excel" or tipo_ficheiro == "xlsx":
        nome_final = f"{nome_ficheiro}.xlsx"
    elif tipo_ficheiro == "txt" or tipo_ficheiro == "texto":
        nome_final = f"{nome_ficheiro}.txt"
    else:
        print("Tipo de ficheiro desconhecido")
        return
    
    #Verifica se o nome do ficheiro já existe
    if os.path.exists(nome_final):
        rescrever = input(f"O ficheiro {nome_final} já existe.\nDeseja substituí-lo? (s/n)").lower()
        if rescrever != "s":
            print("Operação cancelada. Ficheiro não foi guardado")
            return

    #Guarda o ficheiro
    try:
        if tipo_ficheiro == "csv":
            df_save.to_csv(nome_final, index = True)
            print(f"Ficheiro Guardado como: {nome_final}")
        
        elif tipo_ficheiro == "excel" or tipo_ficheiro == "xlsx":
            df_save.to_excel(nome_final, index = True, engine = "openpyxl")
            print(f"Ficheiro Guardado como: {nome_final}")

        elif tipo_ficheiro == "txt" or tipo_ficheiro == "texto":
            df_save.to_csv(nome_final, sep = "\t", index = True)
            print(f"Ficheiro Guardado como: {nome_final}")
            
        
    except ModuleNotFoundError:
        print("ERRO: Falta o módulo openpyxl\nPara instalar correr no terminal: pip install openpyxl")
    
    except PermissionError:
        print("ERRO: Foi obtido um erro de permissão\nVerifica se o ficheiro está aberto.")
    
    except OSError:
        print("ERRO: O nome tem caracteres inválidos")
    
    except Exception as a:
        print(f"ERRO: {a}")
    
def estat(df):
    nome_var = input("Que variável ('Enter' para todas as variáveis)? ").strip()

    def calcs(serie, nome_var):
        #Medidas de Localização 
        media = serie.mean()
        mediana = serie.median()
        moda = serie.mode()
        minimo = serie.min()
        maximo = serie.max()
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)


        print(f"{"-"*35}")
        print(f"Estatísticas de Localização: {nome_var}")
        print(f"\nMédia: {media:.2f}")
        if len(moda) == 1:
            print(f"Moda: {moda.iloc[0]}") #O iloc[0] é para não termoso "lixo" que fica na tela
        else:
            print(f"Moda: {moda.tolist()} (multimoda)")
        print(f"Mínimo: {minimo:.2f}")
        print(f"1º Quartil: {q1:.2f}")
        print(f"Mediana: {mediana:.2f}")
        print(f"3º Quartil: {q3:.2f}")
        print(f"Máximo: {maximo:.2f}")

    
    if not nome_var:
        colunas_num = df.select_dtypes(include = "number").columns
        print(f"Estatísticas de localização para as {len(colunas_num)} variáveis numéricas\n")

        for col in colunas_num:
            calcs(df[col],col)
    
    else:
        if nome_var not in df.columns:
            print(f"{nome_var} não faz parte do conjunto de dados. Escolha outra variável")
            return
        
        serie = df[nome_var]

        if not pd.api.types.is_numeric_dtype(serie):
            print("A variável tem de ser numérica")
            return
        
        calcs(serie,nome_var)

    


def vars(df):
    #Não tenho a certeza se está bem porque ela está a dar algumas de qualidade que são quantitativas 
    print(f"{"Nº":<5} {"Nome da Variável":<30} {"Tipo de Dado"}")
    print("-" * 45)
    for i, coluna in enumerate(df.columns, 1):
        tipo = "Numérica" if pd.api.types.is_numeric_dtype(df[coluna]) else "Categórica"
        print(f"{i:<5}{coluna:<30}{tipo}")




#Menu Global de Visualização (obrigatorio)
def main():
    while True:
        print("\n1 - Visualização geral")
        print("2 - Tipo de Variáveis existentes")
        print("3 - Estatísticas de Localização")
        print("4 - Gráficos?")
        print("5 - Guardar dados em ficheiro")
        print("0 - Sair")

        try:
            opt = int(input("Qual a sua opção? "))
        except ValueError:
            print("Tem de ser um número inteiro entre 0 e 4")

        if opt == 0:
            break
        elif opt == 1:
            print("Ainda sem funcionalidade")
        elif opt == 2:
            vars(df)
        elif opt == 3:
            estat(df)
        elif opt == 4:
            print("Ainda sem função")
        elif opt == 5:
            guardar(df)
        else:
            print("Valor não reconhecido")

if __name__ == "__main__":
    main()
