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
#se não está instalado para as próximas bibliotecas fazer "pip install scikit-learn"
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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
    
def estat():
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

    # A partir daqui é visualização dos qq plots
    resposta = input("\nQuer visualizar o(s) QQ-Plot(s)? (s/n) ").lower().strip()

    if resposta in ["s", "sim", "y", "yes"]:
        
        if len(obj) == 1:
            # Caso 1: Apenas 1 gráfico
            fig, ax = plt.subplots(figsize=(6, 5))
            lista_axes = [ax] # Colocamos numa lista para o loop funcionar igual
        else:
            fig, axes = plt.subplots(3, 3, figsize=(10, 6))
            lista_axes = axes.flatten() #Colocar numa lista que na verdade é um vetor

        # Loop que funciona para os dois casos
        for i, col in enumerate(obj):
            stats.probplot(df[col].dropna(), dist="norm", plot=lista_axes[i])
            
            lista_axes[i].set_title(f"QQ Plot - {col}")
            lista_axes[i].grid(True, alpha=0.3)
            #Estética dos eixos
            lista_axes[i].set_xlabel("Quantis Teóricos")
            lista_axes[i].set_ylabel("Valores Ordenados")

        plt.tight_layout()
        plt.show()

########################
#  Visualização Geral  #
########################

def visualizar_primeiros_10():
    """Mostra os primeiros 10 registos em tabela."""
    
    print("\n" + "="*60)
    print("="+f"{"PRIMEIROS 10 REGISTOS":^58}"+"=")
    print("="*60)
    print(tabulate(df.head(10), headers="keys", tablefmt="github", showindex=True))
    print()

def visualizar_ultimos_10():
    """Mostra os últimos 10 registos em tabela."""
    
    print("\n" + "="*60)
    print("="+f"{"ÚLTIMOS 10 REGISTOS":^58}"+"=")
    print("="*60)
    print(tabulate(df.tail(10), headers="keys", tablefmt="github", showindex=True))
    print()

def visualizar_aleatorios_10():
    """Mostra 10 registos aleatórios em tabela."""
    
    amostra = df.sample(n=10, random_state=None)
    print("\n" + "="*60)
    print("="+f"{"10 REGISTOS ALEATÓRIOS":^58}"+"=")
    print("="*60)
    print(tabulate(amostra, headers="keys", tablefmt="github", showindex=True))
    print()

def visualizar_tabela_completa():
    """Mostra a tabela completa (pode ser grande)."""
    
    print("\n" + "="*60)
    print("="+f"TABELA COMPLETA ({df.shape[0]} registos)".center(58)+"=")
    print("="*60)
    # Para tabelas muito grandes, mostrar em partes
    if df.shape[0] > 50:
        print("⚠ Tabela com muitos registos. Dividida em duas partes:\n")
        print("PARTE 1 (primeiros 25):")
        print(tabulate(df.head(25), headers="keys", tablefmt="github", showindex=True, maxcolwidths=15))
        print("\n...\n")
        print("PARTE 2 (últimos 25):")
        print(tabulate(df.tail(25), headers="keys", tablefmt="github", showindex=True, maxcolwidths=15))
    else:
        print(tabulate(df, headers="keys", tablefmt="github", showindex=True))
    print()

#######################
#         EDA         #
#######################


#Tabelas de frequências
def tabela_frequencias_univariada(coluna):

    try:
        frequencias = df[coluna].value_counts()
        percentagens = df[coluna].value_counts(normalize=True) * 100
        
        tabela = pd.DataFrame({
            "Frequência": frequencias,
            "Percentagem (%)": percentagens.round(2),
            "Frequência Acumulada": frequencias.cumsum(),
            "Percentagem Acumulada (%)": percentagens.cumsum().round(2)
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
        print("="+f"{"TABELAS DE FREQUÊNCIAS":^58}"+"=")
        print("="*60)
        print("1 - Tabela univariada (variável categórica)")
        print("2 - Tabela bivariada (duas variáveis categóricas)")
        print("0 - Voltar")
        print("="*60)
        
        try:
            opt = int(input("Escolha uma opção: "))
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
                        print("\n" + "="*60)
                        print("="+f"FREQUÊNCIAS - {coluna}".center(58)+"=")
                        print("="*60)
                        print(tabulate(tabela, headers="keys", tablefmt="github"))
                        print()

                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")       
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
                        print("\n" + "="*60)
                        print("="+f"TABELA CRUZADA - {col1} vs {col2}".center(58)+"=")
                        print("="*60)
                        print(tabulate(tabela, headers="keys", tablefmt="github"))
                        print()

                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")
                    
            else:
                print("\n✗ Escolha inválida!")
                
        except ValueError:
            print("\nAs escolhas são de 0 a 2!")
            
        
        

#Gráficos univariados

def grafico_univariado_histograma(coluna):
    """
    Cria histograma para coluna numérica.
    """
    try:
        plt.figure(figsize=(10, 5))
        plt.hist(df[coluna], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel(coluna, fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.title(f'Histograma - {coluna}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n✗ Erro ao criar gráfico: {e}")


def grafico_univariado_barras(coluna):
    """
    Cria gráfico de barras para coluna categórica.
    """
    
    try:
        plt.figure(figsize=(10, 5))
        df[coluna].value_counts().plot(kind='bar', color='coral', edgecolor='black', alpha=0.7)
        plt.xlabel(coluna, fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.title(f'Gráfico de Barras - {coluna}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n✗ Erro ao criar gráfico: {e}")


def grafico_univariado_boxplot(coluna):
    """
    Cria boxplot para coluna numérica.
    """
    try:
        plt.figure(figsize=(10, 5))
        plt.boxplot(df[coluna], vert=True)
        plt.ylabel(coluna, fontsize=12)
        plt.title(f'Boxplot - {coluna}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n✗ Erro ao criar gráfico: {e}")


def menu_graficos_univariados():
    """Menu para gráficos univariados."""
    
    colunas_numericas = obter_colunas_por_tipo('numérica')
    colunas_categoricas = obter_colunas_por_tipo('categórica')
    
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

            if opt == 0:
                break
            elif opt == 1:
                print("\nColunas numéricas:")
                for i, col in enumerate(colunas_numericas, 1):
                    print(f"  {i}. {col}")
                
                try:
                    idx = int(input("Escolha o número: ")) - 1
                    coluna = colunas_numericas[idx]
                    grafico_univariado_histograma(coluna)
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")

            elif opt == 2:
                print("\nColunas categóricas:")
                for i, col in enumerate(colunas_categoricas, 1):
                    print(f"  {i}. {col}")
                
                try:
                    idx = int(input("Escolha o número: ")) - 1
                    coluna = colunas_categoricas[idx]
                    grafico_univariado_barras(coluna)
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")

                
            elif opt == 3:
                print("\nColunas numéricas:")
                for i, col in enumerate(colunas_numericas, 1):
                    print(f"  {i}. {col}")
                
                try:
                    idx = int(input("Escolha o número: ")) - 1
                    coluna = colunas_numericas[idx]
                    grafico_univariado_boxplot(coluna)
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")

            else:
                print("\n✗ Escolha inválida!")
        except ValueError:
            print("\nAs escolhas são de 0 a 3!")

        

#Estatísticas por grupo

def menu_estatisticas_grupos():
    """Menu para estatísticas descritivas por grupos."""
    while True:
        colunas_numericas = obter_colunas_por_tipo('numérica')
        colunas_categoricas = obter_colunas_por_tipo('categórica')
        
        print("\nTipos de análise estatística (tipos): ")
        print("1 - Numérica (sem grupos)")
        print("2 - Em grupo (categórica + numérica)")
        print("0 - Voltar")

        try:
            idx_escolha = int(input("Escolha o tipo de análise estatística (número): ")) - 1
            # 1) Voltar atrás    
            if idx_escolha == -1:
                break

            # 2) Estatística simples de uma variável numérica
            elif idx_escolha == 0:
                print("\nColunas numéricas:")
                for i, col in enumerate(colunas_numericas, 1):
                    print(f"  {i}. {col}")
                
                try:
                    idx = int(input("Escolha coluna para análise (número): ")) - 1
                    col = colunas_numericas[idx]
                
                    stats_num = df[col].agg([
                        'count', 'mean', 'std', 'min', 'max', 
                        lambda x: x.quantile(0.25),
                        lambda x: x.quantile(0.50),
                        lambda x: x.quantile(0.75)
                    ])
                    stats_num.index = ['N', 'Média', 'Desvio Padrão', 'Mínimo', 'Máximo', 'Q1', 'Mediana', 'Q3']
                    tabela_stats = stats_num.to_frame(name="Valor")

                    print("\n" + "="*60)
                    print("="+f"ESTATÍSTICAS DE {col.upper()}".center(58)+"=")
                    print("="*60)
                    print(tabulate(tabela_stats, headers="keys", tablefmt="github", floatfmt=".2f"))
                    print()
                        
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")
                
                

            elif idx_escolha == 1:
                print("\nColunas categóricas (agrupamento):")
                for i, col in enumerate(colunas_categoricas, 1):
                    print(f"  {i}. {col}")
                
                try:
                    idx_grupo = int(input("Escolha coluna para agrupamento (número): ")) - 1
                    col_grupo = colunas_categoricas[idx_grupo]
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")

                
                print("\nColunas numéricas (análise):")
                for i, col in enumerate(colunas_numericas, 1):
                    print(f"  {i}. {col}")
                
                try:
                    idx_num = int(input("Escolha coluna para análise (número): ")) - 1
                    col_num = colunas_numericas[idx_num]
                except (ValueError, IndexError):
                    print("\n✗ Escolha inválida!")

                
                try:
                    # Calcular estatísticas
                    stats_gp = df.groupby(col_grupo)[col_num].agg([
                        'count', 'mean', 'std', 'min', 'max', 
                        lambda x: x.quantile(0.25),
                        lambda x: x.quantile(0.50),
                        lambda x: x.quantile(0.75)
                    ])
                    stats_gp.columns = ['N', 'Média', 'Desvio Padrão', 'Mínimo', 'Máximo', 'Q1', 'Mediana', 'Q3']
                    
                    print("\n" + "="*60)
                    print("="+f"ESTATÍSTICAS DE {col_num.upper()} POR {col_grupo.upper()}".center(58)+"=")
                    print("="*60)
                    print(tabulate(stats_gp, headers="keys", tablefmt="github", floatfmt=".2f"))
                    print()

                except Exception as e:
                    print(f"\n✗ Erro ao calcular estatísticas: {e}")
            else:
                print("\n✗ Escolha inválida!")
        except ValueError:
            print("\nAs suas escolhas são de 0 a 2!")

###########################
#         Modelo          #
###########################

def reg_log(df):
    print("A preparar os dados para a construção do modelo")
    #preparação dos dados para podermos aplicar o modelo
    x = df.drop("Sleep_Disorder", axis = 1) #Variáveis independentes
    y = df["Sleep_Disorder"] #Dependente - aquilo que queremos prever

    x = pd.get_dummies(x, columns=["Gender", "Occupation", "BMI_Category"], drop_first=True) #transformar em números

    le = LabelEncoder() # Aqui estamos a transformar as categorias em números
    y = le.fit_transform(y)


    print("Mapeamento das Classes:", dict(zip(le.classes_, le.transform(le.classes_))))
    print("-" * 35)
    '''
    aqui estamos a criar um dicionário que é a nossa cábula para os números:
    o le.classes tem as nossas palavras originais
    o le.transform vai transformar nos números correspondentes
    a função zip formece ao dict os pares key+value
    '''
    #Chunk de treino e teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

    scaler = StandardScaler() #Ajustar as escalas dos valores para não influenciar o modelo

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    modelo = LogisticRegression(max_iter=1000, random_state=123)
    modelo.fit(x_train_scaled,y_train) #treino do modelo
    print("Modelo concluído!")
    
    y_pred = modelo.predict(x_test_scaled) #teste do modelo

    #avaliação do modelo
    print("Relatório de Classificação:\n")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    report = report.replace("precision", "Precisão")
    report = report.replace("recall", "Sensibilidade")
    report = report.replace("f1-score", "F1-Score")
    report = report.replace("support", "Suporte")
    report = report.replace("accuracy", "Acurácia")
    report = report.replace("macro avg", "Média Ma.")
    report = report.replace("weighted avg", "Média Ponde.")
    print(report)
    
    resposta = input("\nQuer visualizar a matriz de confusão?? (s/n) ").lower().strip()

    if resposta in ["s", "sim", "y", "yes"]:
        #matriz de confusão
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="plasma",
                xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title("Matriz de Confusão - Sleep Disorder")
        plt.ylabel('Verdadeiro')
        plt.xlabel('Previsto')
        plt.show()
    else:
        return

###########################
#         Menus           #
###########################

#Menu Visualização Geral 

def menu_visualizaçao_geral():

    while True:
        print("\n" + "="*60)
        print("="+f"{"VISUALIZAÇÃO GERAL":^58}"+"=")
        print("="*60)
        print("1 - Tipo de variáveis existentes")
        print("2 - Primeiros 10 registos")
        print("3 - Últimos 10 registos")
        print("4 - 10 registos aleatórios")
        print("5 - Estatísticas de Localização")
        print("6 - Tabela completa")
        print("0 - Voltar ao menu principal")
        print("="*60)       

        try:
            opt = int(input("Qual a sua opção? "))

            if opt == 0:
                break
            elif opt == 1:
                vars()
            elif opt == 2:
                visualizar_primeiros_10()
            elif opt == 3:
                visualizar_ultimos_10()
            elif opt == 4:
                visualizar_aleatorios_10()
            elif opt == 5:
                estat()
            elif opt == 6:
                visualizar_tabela_completa()
            else:
                print("\n✗ Escolha inválida!")

        except ValueError:
            print("\nAs suas escolhas são de 0 a 6!")


#Menu EDA

def menu_eda():
    
    while True:
        print("\n" + "="*60)
        print("="+f"{"ANÁLISE EXPLORATÓRIA DE DADOS (EDA)":^58}"+"=")
        print("="*60)
        print("1 - Tabelas de Frequências")
        print("2 - Gráficos Univariados")
        print("3 - Gráficos Bivariados")
        print("4 - Estatísticas por Grupos")
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
                print("Fora de serviço!")
            elif opt == 4:
                menu_estatisticas_grupos()
            else:
                print("\n✗ Escolha inválida!")
            
        except ValueError:
            print("\nAs suas escolhas são de 0 a 4!")


#Menu Global de Visualização (obrigatorio)
def main():
    while True:
        print("\n" + "="*60)
        print("="+f"{"ANÁLISE DE DATASET - SLEEP HEALTH AND LIFESTYLE":^58}"+"=")
        print("="*60)
        print("1 - Visualização Geral")
        print("2 - Análise Exploratória de Dados(EDA)")
        print("3 - Avaliação da Normalidade")
        print("4 - Guardar Dados em Ficheiro")
        print("5 - Modelo Logístico")
        print("0 - Sair")
        print("="*60)
        
        try:
            opt = int(input("Qual a sua opção? ").strip())
            if opt == 0:
                print("\nA fechar o programa...\n")
                break
            elif opt == 1:
                menu_visualizaçao_geral()
            elif opt == 2:
                menu_eda()
            elif opt == 3:
                analise_normalidade(df)
            elif opt == 4:
                guardar(df)
            elif opt == 5:
                reg_log(df)
            else:
                print("\n✗ Escolha inválida!")

        except ValueError:
            print("\nAs suas escolhas são de 0 a 5!")


if __name__ == "__main__":
    main()
