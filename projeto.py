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


df = None

def carregar_dados():
    try:
        global df
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


carregar_dados()
print("Occupation types:\n",df["Occupation"].value_counts())


'''
# Verificar se existem valores nulos (muito importante em qualquer projeto)
print("Valores nulos por coluna:\n", df.isnull().sum())

# Ver estatísticas descritivas básicas (Média, min, max, desvio padrão)
print(df.describe())
'''