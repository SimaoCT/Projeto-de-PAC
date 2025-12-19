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

############################
#     Library imports      #
############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
from tabulate import tabulate
from datetime import datetime

###########################
#    Pré processamento    #
########################### 

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

#Limpar nomes de colunas (remover espaços)
df.columns = df.columns.str.replace(" ", "_")


#Tipos de variáveis

tipos_variaveis = {}
  
tipos_variaveis.clear()
     
for i, coluna in enumerate(df.columns, 1):
    tipo = "numérica" if pd.api.types.is_numeric_dtype(df[coluna]) else "categórica"
    tipos_variaveis[coluna] = tipo.lower()



######################
#       Funções      #
######################

#Variáveis númericas
def var_numerica(coluna):
    """Verifica se uma coluna é numérica segundo detecção automática."""
    return tipos_variaveis.get(coluna, 'desconhecida') == 'numérica'

#variáveis categóricas
def var_categorica(coluna):
    """Verifica se uma coluna é categórica segundo detecção automática."""
    return tipos_variaveis.get(coluna, 'desconhecida') == 'categórica'

#Colunas por tipo de variável
def obter_colunas_por_tipo(tipo):

    if tipo == 'numérica':
        return [col for col in df.columns if var_numerica(col)]
    elif tipo == 'categórica':
        return [col for col in df.columns if var_categorica(col)]
    return []

#Guardar em ficheiro (obrigatorio)      

def guardar(df_save):
    tipo_ficheiro = input("\nTipo de formato (ex: csv, txt...): ").lower().strip()
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
        rescrever = input(f"\nO ficheiro {nome_final} já existe.\nDeseja substituí-lo? (s/n)").lower()
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
    nome_var = input("\nQue variável ('Enter' para todas as variáveis)? ").strip()

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

    
def vars():
    print(f'{"\nNº":<5} {"Nome da Variável":<30} {"Tipo de Dado"}')
    print("-" * 45)
    for i, coluna in enumerate(df.columns, 1):
        tipo = tipos_variaveis[coluna].capitalize()  # "numérica" -> "Numérica"
        print(f"{i:<5}{coluna:<30}{tipo}")


def analise_normalidade(df):
    nome_vars = input("\nQual o nome da variável? ('Enter' para todas) ").strip()

    obj = []

    def teste(dados, nome_vars):
        stat, p_value = stats.shapiro(dados)
        alpha = 0.05
        print(f"\n{'-'*35}")
        print(f"Análise de normalidade: {nome_vars}")
        print(f"{'-'*35}")
        print(f"Pressupostos:\nH0: Há normalidade vs H1: Não há normalidade\n")
        print(f"Estatística de teste: {stat:.4f}")
        print(f"p-Value: {p_value:.6f}")

        if p_value > alpha:
            print("Não rejeitamos H0, logo podemos aceitar a hipótese de normalidade")
        else:
            print("Rejeitamos H0, logo rejeitamos a hipótese de normalidade")

    if not nome_vars:
        obj = df.select_dtypes(include = "number").columns
        print(f"Análise de normalidade para as {len(obj)} variáveis numéricas\n")
    
    else:
        if nome_vars not in df.columns:
            print(f"{nome_vars} não faz parte do conjunto de dados. Escolha outra variável")
            return

        if not pd.api.types.is_numeric_dtype(df[nome_vars]):
            print("A variável tem de ser numérica")
            return
        
        obj = [nome_vars]
        
    for col in obj:
        dados = df[col].dropna()
        teste(dados,col)
'''
    #A partir daqui é visualização dos qq plots
    resposta = input("\nQuer visualizar o(s) QQ-Plot(s)? (s/n) ").lower().strip()

    if resposta in ["s", "sim", "y", "yes"]:
        for col in obj:
            dados_limpos = df[col].dropna()
            
            plt.figure(figsize=(6, 4))
            stats.probplot(dados_limpos, dist="norm", plot=plt)
            plt.title(f"QQ Plot - {col}")
            plt.grid(True, alpha=0.3)
            plt.show()

'''

#######################
#         EDA         #
#######################

def tabela_frequencias_univariada(coluna):

    if coluna not in df.columns:
        print(f"\n✗ Coluna '{coluna}' não existe no dataset.\n")
        return None
    
    if not var_categorica(coluna):
        print(f"\n✗ A coluna '{coluna}' é numérica. Tabela de frequências aplica-se apenas a variáveis categóricas.\n")
        return None
    
    try:
        frequencias = df[coluna].value_counts()
        percentagens = df[coluna].value_counts(normalize=True) * 100
        
        tabela = pd.DataFrame({
            "Frequência": frequencias,
            "Percentagem (%)": percentagens.round(2),
            "Frequência Acumulada": frequencias.cumsum(),
            "Porcentagem Acumulada (%)": percentagens.cumsum().round(2)
        })
        
        return tabela
    except Exception as e:
        print(f"\n✗ Erro ao calcular frequências: {e}\n")
        return None


def tabela_frequencias_bivariada(col1, col2):

    if col1 not in df.columns or col2 not in df.columns:
        print(f"\n✗ Uma ou ambas as colunas não existem no dataset.\n")
        return None
    
    if not (var_categorica(col1) and var_categorica(col2)):
        print(f"\n✗ Tabela cruzada requer duas variáveis categóricas!")
        print(f"   - {col1}: {'categórica' if var_categorica(col1) else 'numérica'}")
        print(f"   - {col2}: {'categórica' if var_categorica(col2) else 'numérica'}\n")
        return None
    
    try:
        tabela_cruzada = pd.crosstab(df[col1], df[col2], margins=True)
        return tabela_cruzada
    except Exception as e:
        print(f"\n✗ Erro ao calcular tabela cruzada: {e}\n")
        return None


def menu_tabelas_frequencias():

    colunas_categoricas = obter_colunas_por_tipo('categórica')
    
    while True:
        print("\n" + "="*60)
        print("TABELAS DE FREQUÊNCIAS")
        print("="*60)
        print("1 - Tabela univariada (variável categórica)")
        print("2 - Tabela bivariada (duas variáveis categóricas)")
        print("0 - Voltar")
        print("="*60)
        
        try:
            opt = int(input("Escolha uma opção: "))
        except ValueError:
            print("✗ Entrada inválida!\n")
            continue
        
        if opt == 0:
            break
        elif opt == 1:

            print("\nColunas categóricas disponíveis:")
            for i, col in enumerate(colunas_categoricas, 1):
                print(f"  {i}. {col}")
            
            try:
                idx = int(input("Escolha o número da coluna: ")) - 1
                coluna = colunas_categoricas[idx]
                
                tabela = tabela_frequencias_univariada(coluna)
                if tabela is not None:
                    print("\n" + "="*80)
                    print(f"FREQUÊNCIAS - {coluna} ")
                    print("="*80)
                    print(tabulate(tabela, headers="keys", tablefmt="github"))
                    print()

            except (ValueError, IndexError):
                print("✗ Escolha inválida!\n")
        
        elif opt == 2:
            print("\nColunas categóricas disponíveis:")
            for i, col in enumerate(colunas_categoricas, 1):
                print(f"  {i}. {col}")
            
            try:
                idx1 = int(input("Escolha a 1ª coluna (número): ")) - 1
                idx2 = int(input("Escolha a 2ª coluna (número): ")) - 1
                col1 = colunas_categoricas[idx1]
                col2 = colunas_categoricas[idx2]
                
                tabela = tabela_frequencias_bivariada(col1, col2)
                if tabela is not None:
                    print("\n" + "="*80)
                    print(f"TABELA CRUZADA - {col1} vs {col2}")
                    print("="*80)
                    print(tabulate(tabela, headers="keys", tablefmt="github"))
                    print()

            except (ValueError, IndexError):
                print("✗ Escolha inválida!\n")
        else:
            print("✗ Opção não reconhecida!\n")


###########################
#         Menus           #
###########################


#Menu Vizualisação Geral 

def menu_vizualisaçao_geral():

    while True:
        print("\n" + "="*60)
        print("VIZUALISAÇÃO GERAL")
        print("="*60)
        print("1 - Tipo de variáveis existentes")
        print("2 - Estatísticas de Localização")
        print("3 - Tabela completa")
        print("0 - Voltar ao menu principal")
        print("="*60)
        

        try:
            opt = int(input("Qual a sua opção? "))
        except ValueError:
            print("Tem de ser um número inteiro entre 0 e 3")
            
        if opt == 0:
            break
        elif opt == 1:
            vars()
        elif opt == 2:
            estat(df)
        elif opt == 3:
            print("Ainda sem funcionalidade")
        else:
            print("Valor não reconhecido")

#Menu EDA

def menu_eda():
    
    while True:
        print("\n" + "="*60)
        print("ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
        print("="*60)
        print("1 - Tabelas de Frequências")
        print("2 - Gráficos Univariados")
        print("3 - Gráficos Bivariados")
        print("4 - Estatísticas por Grupos")
        print("0 - Voltar ao menu principal")
        print("="*60)
        
        try:
            opt = int(input("Escolha uma opção: "))
        except ValueError:
            print("✗ Entrada inválida!\n")
            continue
        
        if opt == 0:
            break
        elif opt == 1:
            menu_tabelas_frequencias()
        elif opt == 2:
            print("Fora de serviço!")
        elif opt == 3:
            print("Fora de serviço!")
        elif opt == 4:
            print("Fora de serviço!")
        else:
            print("✗ Opção não reconhecida!\n")

#Menu Global de Visualização (obrigatorio)
def main():
    while True:
        print("\n" + "="*60)
        print("ANÁLISE DE DATASET - SLEEP HEALTH AND LIFESTYLE")
        print("="*60)
        print("1 - Visualização Geral")
        print("2 - Análise Exploratória de Dados(EDA)")
        print("3 - Avaliação da Normalidade")
        print("4 - Guardar Dados em Ficheiro")
        print("0 - Sair")
        print("="*60)
        

        try:
            opt = int(input("Qual a sua opção? "))
        except ValueError:
            print("Tem de ser um número inteiro entre 0 e 4")

        if opt == 0:
            print("\nA fechar o programa...\n")
            break
        elif opt == 1:
            menu_vizualisaçao_geral()
        elif opt == 2:
            menu_eda()
        elif opt == 3:
            analise_normalidade(df)
        elif opt == 4:
            guardar(df)
        else:
            print("Valor não reconhecido")

if __name__ == "__main__":
    main()