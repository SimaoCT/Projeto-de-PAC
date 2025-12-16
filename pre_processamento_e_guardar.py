#Libraries a importar
import pandas as pd
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


def guardar(df_save):
    tipo_ficheiro = input("Tipo de formato (ex: csv, txt...): ").lower()
    nome_ficheiro = input(f"Nome do ficheiro (sem extensão): ")


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

