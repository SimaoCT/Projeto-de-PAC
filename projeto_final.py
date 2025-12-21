'''
Projeto realizado por:
Gonçalo Pato – 114069
João Cordeiro – 114932
Simão Tavares -  113256


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
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


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

#Limpar nomes de colunas (remover espaços)
df.columns = df.columns.str.replace(" ", "_")

#Tipos de variáveis

tipos_variaveis = {}
     
for i, coluna in enumerate(df.columns, 1):
    tipo = "numérica" if pd.api.types.is_numeric_dtype(df[coluna]) else "categórica" #Verificar quais são numéricas ou categóricas
    tipos_variaveis[coluna] = tipo.lower()   #Adiciona ao dicionário a chave, que neste caso é a coluna e o seu valor que é o tipo


#####################
#      Funções      #
#####################


#Variáveis númericas

def var_numerica(coluna):
    return tipos_variaveis.get(coluna, "desconhecida") == "numérica"  #Verifica quais colunas/variáveis são numéricas


#variáveis categóricas

def var_categorica(coluna):
    return tipos_variaveis.get(coluna, "desconhecida") == "categórica" #Verifica quais colunas/variáveis são categóricas


#Colunas por tipo de variável

def obter_colunas_por_tipo(tipo): #Esta função vai retornar uma lista das variáveis consoante o seu tipo

    if tipo == "numérica":
        return [col for col in df.columns if var_numerica(col)] #Dá uma lista de todas as variáveis numéricas
    elif tipo == "categórica":
        return [col for col in df.columns if var_categorica(col)] #Dá uma lista de todas as variáveis categóricas
    return [] #caso o tipo nãos eja nenhum dos outros dois dá uma lista vazia


########################
#  Visualização Geral  #
########################

def variaveis(): #Esta função tem o intuíto de demonstrar as variáveis do dataset e o seu tipo

    print("\n" + "="*60)
    print("=" + f"{'TIPOS DE VARIÁVEIS':^58}" + "=")
    print("="*60)
    print(f"{'Nº':<4} {'Nome da Variável':<29} {'Tipo de Variável'}")
    print("-" * 60)

    for i, coluna in enumerate(df.columns, 1): 
        tipo = tipos_variaveis.get(coluna,"desconhecida").capitalize()  #O capitalize transforma a primeira letra do texto em maiúscula
        print(f"{i:<5}{coluna:<30}{tipo}")


def visualizar_primeiros_10(): #Esta função mostra os primeiros 10 registos do dataset
    
    print("\n" + "="*60)
    print("="+f"{"PRIMEIROS 10 REGISTOS":^58}"+"=")
    print("="*60)
    print(tabulate(df.head(10), headers="keys", tablefmt="github", showindex=True)) #Cria uma tabela com os nomes das colunas sendo as "keys"(Variáveis)
    print()                                                                         


def visualizar_ultimos_10(): #Esta função mostra os últimos 10 registos do dataset
    
    print("\n" + "="*60)
    print("="+f"{"ÚLTIMOS 10 REGISTOS":^58}"+"=")
    print("="*60)
    print(tabulate(df.tail(10), headers="keys", tablefmt="github", showindex=True))
    print()


def visualizar_aleatorios_10(): #Esta função mostra 10 registos aleatórios do dataset
    
    amostra = df.sample(n=10,random_state=None)
    print("\n" + "="*60)
    print("="+f"{"10 REGISTOS ALEATÓRIOS":^58}"+"=")
    print("="*60)
    print(tabulate(amostra, headers="keys", tablefmt="github", showindex=True))
    print()


def visualizar_tabela_completa(): #Esta função mostra uma tabela completa de dados sendo o seu máximo 50 registos
    
    print("\n" + "="*60)
    print("="+f"TABELA COMPLETA ({df.shape[0]} registos)".center(58)+"=")  #O center permite centrar o texto
    print("="*60)
    # Para tabelas muito grandes, mostra em duas partes
    if df.shape[0] > 50:  #Verifica se a tabela tem mais de 50 linhas
        print("Tabela com muitos registo! Dividida em duas partes:\n")
        print("PARTE 1 (primeiros 25):")
        print(tabulate(df.head(25), headers="keys", tablefmt="github", showindex=True))
        print("\n...\n")
        print("PARTE 2 (últimos 25):")
        print(tabulate(df.tail(25), headers="keys", tablefmt="github", showindex=True))
    else:
        print(tabulate(df, headers="keys", tablefmt="github", showindex=True))
    print()


#######################
#         EDA         #
#######################

#Tabelas de frequências

def tabela_frequencias_univariada(coluna): #Esta função cria uma tabela com frequências absolutas e relativas e também as frequências acumuladas

    try:
        frequencias = df[coluna].value_counts() #Conta quantas vezes cada valor aparece na variável
        percentagens = df[coluna].value_counts(normalize=True)*100 #Em vez de retornar a contagem, devolve a proporção que é multiplicada por 100 para ter a percentagem
        
        tabela = pd.DataFrame({
            "Frequência": frequencias,
            "Percentagem (%)": percentagens.round(2),
            "Frequência Acumulada": frequencias.cumsum(),
            "Percentagem Acumulada (%)": percentagens.cumsum().round(2)
        }) #Cria um dataframe para a análise descritiva
        
        return tabela
    except Exception as erro: #Encontra qualquer erro durante a execução da função
        print(f"\n✗ Erro ao calcular frequências: {erro}\n")
        return None


def tabela_frequencias_bivariada(col1, col2): #Esta função cria uma tabela de contingência entre duas variáveis
    
    try:
        tabela_cruzada = pd.crosstab(df[col1], df[col2], margins=True) #Cria a tabela de contingência
        return tabela_cruzada
    except Exception as erro:
        print(f"\n✗ Erro ao calcular tabela cruzada: {erro}\n")
        return None


def menu_tabelas_frequencias(): #Menu interativo das tabelas de frequência

    colunas_categoricas = obter_colunas_por_tipo("categórica") #Para obter a lista com todas as variáveis categóricas
    
    while True:  #O menu aparece sempre até a pessoa querer sair (0)
        print("\n" + "="*60)
        print("="+f"{"TABELAS DE FREQUÊNCIAS":^58}"+"=")
        print("="*60)
        print("1 - Tabela univariada (variável categórica)")
        print("2 - Tabela bivariada (duas variáveis categóricas)")
        print("0 - Voltar")
        print("="*60)
        
        try:
            opt = int(input("Escolha uma opção: ")) #Para escolher quais das opções mostradas no menu
            if opt == 0: #Caso a opção escolhida seja o 0 este vai voltar ao menu anterior
                break
            elif opt == 1: #Irá passar para a formação da tabela univariada
                print("\nVariáveis categóricas disponíveis:")
                for i, col in enumerate(colunas_categoricas, 1): #Vai dar print a todas as variáveis categóricas de forma a tornar mais fácil a escolha para as tabelas
                    print(f"  {i}. {col}")
                
                try:
                    idx = int(input("Escolha a variável (número): ")) - 1 #Vai dar o index da coluna da variável
                    coluna = colunas_categoricas[idx] #Varíavel escolhida
                    
                    tabela = tabela_frequencias_univariada(coluna) #Tabela formada da variável escolhida
                    if tabela is not None: #Vai imprimir a tabela
                        print("\n" + "="*60)
                        print("="+f"FREQUÊNCIAS - {coluna}".center(58)+"=")
                        print("="*60)
                        print(tabulate(tabela, headers="keys", tablefmt="github"))
                        print()

                except (ValueError, IndexError): #Se não for nenhuma das opções irá dar erro
                    print("\n✗ Escolha inválida!")
                        
            elif opt == 2: #Ira passar para a formação da tabela de  contingencia
                print("\nVariáveis categóricas disponíveis:")
                for i, col in enumerate(colunas_categoricas, 1):
                    print(f"  {i}. {col}")
                
                try:
                    idx1 = int(input("Escolha a 1ª variável (número): ")) - 1
                    idx2 = int(input("Escolha a 2ª variável (número): ")) - 1
                    col1 = colunas_categoricas[idx1]
                    col2 = colunas_categoricas[idx2]
                    
                    tabela = tabela_frequencias_bivariada(col1, col2)
                    if tabela is not None:
                        print("\n" + "="*60)
                        print("="+f"TABELA CRUZADA - {col1} vs {col2}".center(58)+"=")
                        print("="*60)
                        print(tabulate(tabela, headers="keys", tablefmt="github"))
                        print()#Este print está aqui a fzer o q?

                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")
                    
            else: #Caso o número da opção escolhido fora dos parametros definidos
                print("\n✗ Escolha inválida!")
                
        except ValueError: #Caso nem escolham um número e seja anything
            print("\nAs escolhas são de 0 a 2!")


#Gráficos univariados

def grafico_univariado_histograma(coluna): #Esta função tem como objetivo criar um histograma
    try:
        plt.figure(figsize=(10, 5))  #Vai defenir o tamanho da figura neste caso 10 de largura e 5 de altura
        plt.hist(df[coluna], bins=30, color="steelblue", edgecolor="black", alpha=0.7) #Cria o histograma com 30 intervalos
        plt.xlabel(coluna, fontsize=12) #Dá o nome da variável ao eixo  dos x
        plt.ylabel("Frequência", fontsize=12) 
        plt.title(f"Histograma - {coluna}", fontsize=14, fontweight="bold") 
        plt.grid(True, alpha=0.3) # grelha
        plt.tight_layout() #Serve para ajustar as margens auto
        plt.show() 

    except Exception as erro:
        print(f"\n✗ Erro ao criar gráfico: {erro}")


def grafico_univariado_barras(coluna): #Esta funcao tem como objetivo criar um grafico de barras
    
    try:
        plt.figure(figsize=(10, 5))
        df[coluna].value_counts().plot(kind="bar", color="coral", edgecolor="black", alpha=0.7) #Vai criar o gráfico de barras contando de  cada vez que cada categoria aparece
        plt.xlabel(coluna, fontsize=12)
        plt.ylabel("Frequência", fontsize=12)
        plt.title(f"Gráfico de Barras - {coluna}", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right") #Vai fazer com que os nomes das categorias girem 45 para evitar sobreposição e ficarem alinahdos à direita
        plt.grid(True, alpha=0.3, axis="y") #Vai ser aplicada apenas ao y
        plt.tight_layout()
        plt.show()

    except Exception as erro:
        print(f"\n✗ Erro ao criar gráfico: {erro}")


def grafico_univariado_boxplot(coluna): #Esta função tem como objetivo criar um boxplot

    try:
        plt.figure(figsize=(10, 5))
        plt.boxplot(df[coluna], vert=True)
        plt.ylabel(coluna, fontsize=12)
        plt.title(f"Boxplot - {coluna}", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()

    except Exception as erro:
        print(f"\n✗ Erro ao criar gráfico: {erro}")


def menu_graficos_univariados(): #Menu dos gráficos univariados
    
    colunas_numericas = obter_colunas_por_tipo("numérica") #Obter a lista com todas as variaveis numéricas
    colunas_categoricas = obter_colunas_por_tipo("categórica") # Obter a lista com todas as variáveis categóricas
    
    while True:
        print("\n" + "="*60)
        print("="+f"{"GRÁFICOS UNIVARIADOS":^58}"+"=")
        print("="*60)
        print("1 - Histograma (variável numérica)")
        print("2 - Gráfico de Barras (variável categórica)")
        print("3 - Boxplot (variável numérica)")
        print("0 - Voltar")
        print("="*60)
        
        try:
            opt = int(input("Escolha uma opção: "))

            if opt == 0: #Opção escolhida para voltar para tras
                break
            elif opt == 1: #Opção escolhida para os histogramas
                print("\nVariáveis numéricas disponíveis:")
                for i, col in enumerate(colunas_numericas, 1): #Printa as opções
                    print(f"  {i}. {col}")
                
                try:
                    idx = int(input("Escolha a variável (número): ")) - 1
                    coluna = colunas_numericas[idx] #Variável numérica escolhida
                    grafico_univariado_histograma(coluna) #Função do histograma
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")

            elif opt == 2: #Opção escolhida para os gráficos de barras
                print("\nVariáveis categóricas disponíveis:")
                for i, col in enumerate(colunas_categoricas, 1): #Dá print das opções
                    print(f"  {i}. {col}")
                
                try:
                    idx = int(input("Escolha a variável (número): ")) - 1
                    coluna = colunas_categoricas[idx] #variável categórica escolhida
                    grafico_univariado_barras(coluna) #Função do gráfico de barras
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")

                
            elif opt == 3: #Opção escolhida para os boxplots
                print("\nVariáveis numéricas disponíveis:")
                for i, col in enumerate(colunas_numericas, 1): # Printa as opções
                    print(f"  {i}. {col}")
                
                try:
                    idx = int(input("Escolha a variável (número): ")) - 1
                    coluna = colunas_numericas[idx] #Variável numérica escolhida
                    grafico_univariado_boxplot(coluna) #Função do boxplot
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")

            else:
                print("\n✗ Escolha inválida!")
        except ValueError:
            print("\nAs escolhas são de 0 a 3!")


#Gráficos bivariados

def grafico_bivariado_scatter(col_x, col_y, col_cor=None): #Esta função tem como objetivo criar um gráfico de dispersão entre duas variáveis

    try:
        plt.figure(figsize=(10, 6))

        if col_cor is not None:
            categorias = df[col_cor].astype(str) #converte a variável categórica em string 
            categorias_unicas = categorias.unique() #Vai obter as categorias únicas da variável
            palette = sns.color_palette("tab10", len(categorias_unicas)) # vai criar uma paleta de cores para o número de categorias únicas
            cores = dict(zip(categorias_unicas, palette)) #Vai criar um dicionário que associa cada categoria a uma cor por cada categoria

            for cat in categorias_unicas:
                mask = categorias == cat #Vai indicar quais linhas do dataframe pertencem à categoria cat
                plt.scatter(
                    df.loc[mask, col_x], #Vai selecionar os pontos da variável do eixo X tendo em conta a mask
                    df.loc[mask, col_y], #Vai selecionar os pontos da variável do eixo Y tendo em conta a mask
                    label=cat, #Dá o nome da categoria na legenda
                    color=cores[cat], #vai dar a cor associada a cada categoria já definina anteriormente
                    alpha=0.7,
                    edgecolor="black"
                )
            plt.legend(title=col_cor, bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.scatter(df[col_x], df[col_y], color="steelblue", alpha=0.7, edgecolor="black") #Se a col_cor continuar None os pontos ficam todos com a mesma cor

        plt.xlabel(col_x, fontsize=12)
        plt.ylabel(col_y, fontsize=12)
        plt.title(f"Scatter Plot - {col_x} vs {col_y}", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as erro:
        print(f"\n✗ Erro ao criar scatter plot: {erro}")


def grafico_bivariado_boxplot(col_cat, col_num): #Esta função tem como objetivo criar um boxplot de uma variável numérica em funcção de uma categórica

    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col_cat], y=df[col_num])
        plt.xlabel(col_cat, fontsize=12)
        plt.ylabel(col_num, fontsize=12)
        plt.title(f"Boxplot - {col_num} por {col_cat}", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()
    except Exception as erro:
        print(f"\n✗ Erro ao criar boxplot bivariado: {erro}")


def grafico_bivariado_heatmap(col1, col2): #Esta função tem como objetivo criar um heatmap entre duas variaveis categóricas
    try:
        tabela = pd.crosstab(df[col1], df[col2]) #Vai criar uma tabela  entre as duas variáveis categóricas

        plt.figure(figsize=(10, 6))
        sns.heatmap(tabela, annot=True, fmt="d", cmap="YlOrRd") #annot -> os valores estão dentro de cada célula; fmt, os números aparecem como inteiros; cmap, paleta de cores
        plt.title(f"Heatmap - {col1} vs {col2}", fontsize=14, fontweight="bold")
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.tight_layout()
        plt.show()

    except Exception as erro:
        print(f"\n✗ Erro ao criar heatmap: {erro}")


def menu_graficos_bivariados(): #Menu dos graficos bivariados
    colunas_numericas = obter_colunas_por_tipo("numérica")
    colunas_categoricas = obter_colunas_por_tipo("categórica")

    while True:
        print("\n" + "="*60)
        print("=" + f"{'GRÁFICOS BIVARIADOS':^58}" + "=")
        print("="*60)
        print("1 - Scatter Plot (2 numéricas, opcionalmente colorido por categórica)")
        print("2 - Boxplot (numérica por categórica)")
        print("3 - Heatmap (2 categóricas)")
        print("0 - Voltar")
        print("="*60)


        try:
            opt = int(input("Escolha uma opção: "))

            if opt == 0: # Opção escolhida voltar para trás
                break

            elif opt == 1: #Opção escolhida para os gráficos de dispersão
                print("\nVariáveis numéricas disponíveis:")
                for i, col in enumerate(colunas_numericas, 1): #Dá print das opções
                    print(f"  {i}. {col}")

                try:
                    idx_x = int(input("Escolha a variável para o eixo X (número): ")) - 1
                    idx_y = int(input("Escolha a variável para o eixo Y (número): ")) - 1
                    col_x = colunas_numericas[idx_x] #Variável numérica escolhida para o eixo X
                    col_y = colunas_numericas[idx_y] # Variável numérica escolhida para o eixo Y
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")
                    continue

                col_cor = None
                if colunas_categoricas: #Caso  se queria agrupar por variável categórica
                    print("\nVariáveis categóricas (para cor):")
                    print("  0. Sem cor (não agrupar)")
                    for i, col in enumerate(colunas_categoricas, 1): #Printa as opções
                        print(f"  {i}. {col}")
                    try:
                        idx_cor = int(input("Escolha variável categórica para cor (0 para nenhuma): ")) # Escolha se querremos agrupar por uma variável categorica  
                        if idx_cor > 0:
                            col_cor = colunas_categoricas[idx_cor - 1]
                        elif idx_cor == 0:
                            col_cor = None

                        grafico_bivariado_scatter(col_x, col_y, col_cor)
                    except (ValueError, IndexError):
                        print("\n✗ Escolha inválida!")
                        continue

            elif opt == 2: # Opção escolhida para os boxplots

                print("\nVariáveis categóricas (agrupamento):")
                for i, col in enumerate(colunas_categoricas, 1): #Printa as opçõess
                    print(f"  {i}. {col}")
                try:
                    idx_cat = int(input("Escolha a variável categórica (número): ")) - 1
                    col_cat = colunas_categoricas[idx_cat] #Variável categórica escolhida

                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")
                    continue

                
                print("\nVariáveis numéricas disponíveis:")
                for i, col in enumerate(colunas_numericas, 1):  #Dá print das opções
                    print(f"  {i}. {col}")
                try:
                    idx_num = int(input("Escolha a variável numérica (número): ")) - 1
                    col_num = colunas_numericas[idx_num]    #Variavel numérica escolhida
                    grafico_bivariado_boxplot(col_cat, col_num)
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")
                    continue
                
            elif opt == 3: #Opção escolhida para os heatmaps
                print("\nVariáveis categóricas disponíveis:")
                for i, col in enumerate(colunas_categoricas, 1): #Printa as opções
                    print(f"  {i}. {col}")

                try:
                    idx1 = int(input("Escolha a 1ª variável (número): ")) - 1
                    idx2 = int(input("Escolha a 2ª variável (número): ")) - 1
                    col1 = colunas_categoricas[idx1] #Primeira variável categórica escolhida
                    col2 = colunas_categoricas[idx2] # Segunda variável categórica escolhida
                    grafico_bivariado_heatmap(col1, col2) #chama a função do heatmap
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")
                    continue

            else:
                print("\n✗ Escolha inválida!")
           
        except ValueError:
            print("\nAs escolhas são de 0 a 3!")
            continue


#Estatísticas por grupo

def menu_estatisticas_grupos(): #Menu interativo das estatísticas 

    while True: 
        colunas_numericas = obter_colunas_por_tipo("numérica") #Para obter a lista com todas as variáveis numéricas
        colunas_categoricas = obter_colunas_por_tipo("categórica") # Para obter a lista com todas as variáveis categóricas

        print("\nTipos de análise estatística (tipos): ")
        print("1 - Numérica (sem grupos)")
        print("2 - Em grupo (categórica + numérica)")
        print("0 - Voltar")

        try:
            idx_escolha = int(input("Escolha uma opção: ")) - 1
    
            if idx_escolha == -1: #Opção escolhida para voltar para trás
                break

            elif idx_escolha == 0: #Opção escolhida para as estatísticas numéricas sem grupos
                print("\nVariáveis numéricas disponíveis:")
                for i, col in enumerate(colunas_numericas, 1): #Dá print das opções
                    print(f"  {i}. {col}")
                
                try:
                    idx = int(input("Escolha a variável para análise (número): ")) - 1
                    col = colunas_numericas[idx] #Variável numérica escolhida
                
                    estatisticas_num = df[col].agg([
                        "count", "mean", "std", "min", "max",
                        lambda x: x.quantile(0.25),
                        lambda x: x.quantile(0.50),
                        lambda x: x.quantile(0.75)
                    ]) #calculo das estatísticas
                    estatisticas_num.index = ["N", "Média", "Desvio Padrão", "Mínimo", "Máximo", "Q1", "Mediana", "Q3"] #Dá estes nomes ao índice
                    tabela_estatisticas = estatisticas_num.to_frame(name="Valor") #Converte a série  para dataframe

                    print("\n" + "="*60)
                    print("="+f"ESTATÍSTICAS DE {col.upper()}".center(58)+"=")
                    print("="*60)
                    print(tabulate(tabela_estatisticas, headers="keys", tablefmt="github", floatfmt=".2f"))
                    print()
                        
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")
                
                

            elif idx_escolha == 1: # Opção escolhida para as estatísticas em grupo
                print("\nVariáveis categóricas (agrupamento):")
                for i, col in enumerate(colunas_categoricas, 1): #Printa as opcoes
                    print(f"  {i}. {col}")
                
                try:
                    idx_grupo = int(input("Escolha a variável para agrupamento (número): ")) - 1
                    col_grupo = colunas_categoricas[idx_grupo] #Variável categórica escolhida
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")


                print("\nVariáveis numéricas (análise):") 
                for i, col in enumerate(colunas_numericas, 1): #Dá print das opções
                    print(f"  {i}. {col}")
                
                try:
                    idx_num = int(input("Escolha a variável para análise (número): ")) - 1
                    col_num = colunas_numericas[idx_num] #Variável numerica escolhida
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")

                
                try:
                    estatisticas_gp = df.groupby(col_grupo)[col_num].agg([
                        "count", "mean", "std", "min", "max", 
                        lambda x: x.quantile(0.25),
                        lambda x: x.quantile(0.50),
                        lambda x: x.quantile(0.75)
                    ]) # Calcular as estatísticas por grupo
                    estatisticas_gp.columns = ["N", "Média", "Desvio Padrão", "Mínimo", "Máximo", "Q1", "Mediana", "Q3"] #Dá estes nomes às colunas
                    
                    print("\n" + "="*60)
                    print("="+f"ESTATÍSTICAS DE {col_num.upper()} POR {col_grupo.upper()}".center(58)+"=")
                    print("="*60)
                    print(tabulate(estatisticas_gp, headers="keys", tablefmt="github", floatfmt=".2f")) 
                    print()

                except Exception as erro:
                    print(f"\n✗ Erro ao calcular estatísticas: {erro}")
            else:
                print("\n✗ Escolha inválida!")
        except ValueError:
            print("\nAs suas escolhas são de 0 a 2!")


# Análise da normalidade

def analise_normalidade():

    colunas_numericas = obter_colunas_por_tipo("numérica")
    while True:
        #Aqui e print do micro menuu
        print("\nVariáveis disponíveis:")
        opt_todas = len(colunas_numericas)+1
        for i, col in enumerate(colunas_numericas, 1): #Dá print das opções
            print(f"  {i}. {col}")
        print(f" {opt_todas}. Todas as Variáveis")
        print("  0. Voltar") #Opção para voltar para trás

        try:  
            idx_num = int(input("\nEscolha uma variável: ")) - 1
            
            if idx_num == -1: #Opção escolhida para voltar para trás
                break

            obj = []
            if idx_num == len(colunas_numericas):
                obj = colunas_numericas
                #Esta parte aqui é que nos  permite ver todas as variaveias ao mesmo tempo
            else:
                col = colunas_numericas[idx_num] #Variável numérica escolhida2
                obj = [col]

            
            def teste(dados, col): #Função que realiza o teste de Shapiro-Wilk às variáveis numéricas
                estatistica, p_value = stats.shapiro(dados)  #Teste de Shapiro-Wilk
                alpha = 0.05 #Nível de significância do teste

                print("\n" + "="*60)
                print("="+f"Análise de normalidade: {col}".center(58)+"=")
                print("="*60)
                print(f"Pressupostos:\nH0: Há normalidade vs H1: Não há normalidade\n")
                print(f"Estatística de teste: {estatistica:.4f}")
                print(f"p-Value: {p_value:.6f}")

                if p_value > alpha: 
                    print("Não rejeitamos H0, logo podemos aceitar a hipótese de normalidade")
                else:
                    print("Rejeitamos H0, logo rejeitamos a hipótese de normalidade")             
                
            for x in obj:
                dados = df[x].dropna()
                teste(dados,x)

            # A partir daqui é visualização dos qq plots
            resposta = input("\nQuer visualizar o(s) QQ-Plot(s)? (s/n) ").lower().strip()

            if resposta in ["s", "sim", "y", "yes"]:
                
                if len(obj) == 1:
                    #Partir a figura em subplots este é para a figura única
                    fig, ax = plt.subplots(figsize=(6, 5))
                    lista_axes = [ax] 
                else:
                    #Aqui é quando  temos tudo
                    fig, axes = plt.subplots(3, 3, figsize=(10, 6))
                    lista_axes = axes.flatten() 


                for i, col_n in enumerate(obj):
                    stats.probplot(df[col_n].dropna(), dist="norm", plot=lista_axes[i])
                    
                    lista_axes[i].set_title(f"QQ Plot - {col_n}")
                    lista_axes[i].grid(True, alpha=0.3)
                    lista_axes[i].set_xlabel("Quantis Teóricos")
                    lista_axes[i].set_ylabel("Valores Ordenados")

                plt.tight_layout()
                plt.show()
        except (ValueError, IndexError):
            print("\n✗ Escolha inválida!")


###########################
#         Modelo          #
###########################

def reg_log(df):
    print("\nA preparar os dados para a construção do modelo...")
    #preparação dos dados para podermos aplicar o modelo
    x = df.drop("Sleep_Disorder", axis = 1) #Variáveis independentes
    y = df["Sleep_Disorder"] #Dependente - aquilo que queremos prever

    x = pd.get_dummies(x, columns=["Gender", "Occupation", "BMI_Category"], drop_first=True) #transformar  em números

    le = LabelEncoder() # Aqui estamos a transformar as categoria em números
    y = le.fit_transform(y)

    print("\n" + "="*60)
    print("="+f"{"MODELO":^58}"+"=")
    print("=" * 60)
    print("Mapeamento das Classes:", dict(zip(le.classes_, le.transform(le.classes_))))#Esta linha é importante forma um dict automátcio os os nomes e números atribuídos

    '''
    aqui estamos a criar um dicionário que é a nossa cábula para os números:
    o le.classes tem as nossas palavras originais
    o le.transform vai transformar nos números correspondentes
    a função zip formece ao dict os pares key+value
    '''
    #Chunk de treino e teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

    scaler = StandardScaler() #Ajustar a escalas dos valores  para não influenciar o modelo


    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    modelo = LogisticRegression(max_iter=1000, random_state=123)
    modelo.fit(x_train_scaled,y_train) #treino do modelo
    print("\nModelo concluído!")
    
    y_pred = modelo.predict(x_test_scaled) #teste do modelo

    #avaliação do modelo
    print("Relatório de Classificação:\n")
    report = classification_report(y_test, y_pred,target_names=le.classes_)
    report = report.replace("precision", "Precisão")
    report = report.replace("recall", "Sensibilidade")
    report = report.replace("f1-score", "F1-Score")
    report = report.replace("support","Suporte")
    report = report.replace("accuracy", "Acurácia")
    report = report.replace("macro avg","Média Ma.")
    report = report.replace("weighted avg", "Média Ponde.")
    print(report)
    
    resposta = input("\nQuer visualizar a matriz de confusão? (s/n) ").lower().strip()

    if resposta in ["s", "sim", "y", "yes"]:
        #matriz  de confusão
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="plasma",
                xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title("Matriz de Confusão - Sleep Disorder")
        plt.ylabel("Verdadeiro")
        plt.xlabel("Previsto")
        plt.show()
    else:
        return


############################
#     Guardar Ficheiro     #
############################    

def guardar():

    print("\n" + "="*60)
    print("="+f"{"GUARDAR FICHEIRO":^58}"+"=")
    print("=" * 60)
    tipo_ficheiro = input("\nTipo de formato (ex: csv, txt, excel): ").lower().strip()
    nome_ficheiro = input(f"Nome do ficheiro (sem extensão): ").strip()


    #Verifica o  tipo de ficheiro
    if tipo_ficheiro == "csv":
        nome_final = f"{nome_ficheiro}.csv"
    elif tipo_ficheiro == "excel" or tipo_ficheiro == "xlsx":
        nome_final = f"{nome_ficheiro}.xlsx"
    elif tipo_ficheiro == "txt" or tipo_ficheiro == "texto":
        nome_final = f"{nome_ficheiro}.txt"
    else:
        print("!Operação cancelada. Tipo de ficheiro desconhecido!")
        return
    
    #Verifica se o nome do ficheiro já existe
    if os.path.exists(nome_final):
        rescrever = input(f"\nO ficheiro {nome_final} já existe.\nDeseja substituí-lo? (s/n) ").lower()
        if rescrever not in ["s","y","yes","sim"]:
            print("!Operação cancelada. Ficheiro não foi guardado!")
            return

    #Guarda o ficheiro
    try:
        if tipo_ficheiro == "csv":
            df.to_csv(nome_final, index = True)
            print(f"Ficheiro Guardado como: {nome_final}")
        
        elif tipo_ficheiro == "excel" or tipo_ficheiro == "xlsx":
            df.to_excel(nome_final, index = True, engine = "openpyxl")
            print(f"Ficheiro Guardado como: {nome_final}")

        elif tipo_ficheiro == "txt" or tipo_ficheiro == "texto":
            df.to_csv(nome_final, sep = "\t", index = True)
            print(f"Ficheiro Guardado como: {nome_final}")
            
        
    except ModuleNotFoundError:
        print("ERRO: Falta o módulo openpyxl\nPara instalar correr no terminal: pip install openpyxl")
    
    except PermissionError:
        print("ERRO: Foi obtido um erro de permissão\nVerifica se o ficheiro está aberto.")
    
    except OSError:
        print("ERRO: O nome tem caracteres inválidos")
    
    except Exception as erro:
        print(f"ERRO: {erro}")


###########################
#         Menus           #
###########################

#Menu Visualização Geral 

def menu_visualizaçao_geral(): #Menu interativo da visualização geral

    while True:
        print("\n" + "="*60)
        print("="+f"{"VISUALIZAÇÃO GERAL":^58}"+"=")
        print("="*60)
        print("1 - Tipo de variáveis existentes")
        print("2 - Primeiros 10 registos")
        print("3 - Últimos 10 registos")
        print("4 - 10 registos aleatórios")
        print("5 - Tabela completa")
        print("0 - Voltar ao menu principal")
        print("="*60)       

        try:
            opt = int(input("Escolha uma opção: "))

            if opt == 0:
                break
            elif opt == 1:
                variaveis()
            elif opt == 2: 
                visualizar_primeiros_10()
            elif opt == 3:
                visualizar_ultimos_10()
            elif opt == 4:
                visualizar_aleatorios_10()
            elif opt == 5:
                visualizar_tabela_completa()
            else:
                print("\n✗ Escolha inválida!")

        except ValueError:
            print("\nAs suas escolhas são de 0 a 5!")


#Menu EDA

def menu_eda(): #Menu iterativo  da análise exploratória de dados
    
    while True:
        print("\n" + "="*60)
        print("="+f"{"ANÁLISE EXPLORATÓRIA DE DADOS (EDA)":^58}"+"=")
        print("="*60)
        print("1 - Tabelas de Frequências")
        print("2 - Gráficos Univariados")
        print("3 - Gráficos Bivariados")
        print("4 - Estatísticas por Grupos")
        print("5 - Avaliação da Normalidade")
        print("0 - Voltar ao menu principal")
        print("="*60)
        
        try:
            opt = int(input("Escolha uma opção: "))
            if opt == 0:
                break
            elif opt == 1:
                menu_tabelas_frequencias()
            elif opt == 2:
                menu_graficos_univariados()
            elif opt == 3:
                menu_graficos_bivariados()
            elif opt == 4:
                menu_estatisticas_grupos()
            elif opt == 5:
                analise_normalidade()
            else:
                print("\n✗ Escolha inválida!")
            

        except ValueError:
            print("\nAs suas escolhas são de 0 a 5!")


#Menu Global de Visualização (obrigatorio)
def main(): #Menu interativo principal
    while True:
        print("\n" + "="*60)
        print("="+f"{"ANÁLISE DE DATASET - SLEEP HEALTH AND LIFESTYLE":^58}"+"=")
        print("="*60)
        print("1 - Visualização Geral")
        print("2 - Análise Exploratória de Dados(EDA)")
        print("3 - Modelo")
        print("4 - Guardar Dados em Ficheiro")
        print("0 - Sair")
        print("="*60)
        
        try:
            opt = int(input("Escolha uma opção: "))
            if opt == 0:
                print("\nA fechar o programa...\n")
                print("Obrigado pelo seu tempo! Até à próxima :)\n")
                break
            elif opt == 1:
                menu_visualizaçao_geral()
            elif opt == 2:
                menu_eda()
            elif opt == 3:
                reg_log(df)            
            elif opt == 4:
                guardar()
            else:
                print("\n✗ Escolha inválida!")

        except ValueError:
            print("\nAs suas escolhas são de 0 a 4!")


if __name__ == "__main__":
    main()

