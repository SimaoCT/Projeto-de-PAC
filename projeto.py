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


df = None

def carregar_dados():
    try:
        global df
        #a = str(input("Qual o nome do ficehrio? "))
        #b = str(input("Qual o tipo de ficheiro? "))1
        df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv", encoding= "utf-8")
        #Definir o índice como o Person ID
        df = df.set_index("Person ID")
        # Separar a coluna "Blood Pressure" em duas novas colunas numéricas
        df[["BP_Sys","BP_Dias"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
        #Visualizar os valores nulos
        print("Valores nulos:\n",df.isnull().sum())

        df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")


    except FileNotFoundError:
        print("Verifica se o ficheiro está na mesma pasta ou se o diretório é o correto")




#print("Occupation types:\n",df["Occupation"].value_counts())

'''

# Ver estatísticas descritivas básicas (Média, min, max, desvio padrão)
print(df.describe())


# 1. Define your data
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 50]

# 2. Plot the data
mp.plot(x, y, color='green', marker='o', linestyle='dashed')

# 3. Add labels
mp.title("O meu  título")
mp.xlabel("AAAAAAAAAAAAAAAAAAA")
mp.ylabel("Y Axis")

# 4. Show the plot


# 1. Load an example dataset provided by Seaborn
df = sns.load_dataset("tips")

# 2. Create the plot
# 'hue' colors the dots based on a category (e.g., smoker vs non-smoker)
sns.scatterplot(data=df, x="total_bill", y="tip", hue="sex")

# 3. Show the plot
mp.title("Tips vs Total Bill")
mp.show()

'''

def main():
    while True:
        print("\n1 - Entrada dos dados")
        print("2 - Imprime dados")
        print("3 - Guardar dados em ficheiro")
        print("4 - ")
        print("0 - Sair")

        try:
            opt = int(input("Qual a sua opção? "))
        except ValueError:
            print("Tem de ser um número inteiro entre 0 e 4")

        if opt == 0:
            break
        if opt == 1:
            #Lê os dados a partir do ficheiro 
            carregar_dados()

main()
