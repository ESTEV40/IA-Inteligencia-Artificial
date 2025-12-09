import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. CARREGAR DADOS
try:
    df = pd.read_csv("resultados.csv")
except FileNotFoundError:
    print("ERRO: Arquivo 'resultados.csv' não encontrado.")
    exit()

df.columns = df.columns.str.strip()

# Garante que a coluna de repetições é número
df['repeticoes'] = pd.to_numeric(df['repeticoes'], errors='coerce')

# Remove espaços dos nomes dos testes (ex: "Elitismo " vira "Elitismo")
df['teste'] = df['teste'].astype(str).str.strip()

# Configuração de Estilo
plt.style.use('ggplot')

# Filtra apenas onde repetições == 100
df_100 = df[df['repeticoes'] == 100].copy() # .copy() evita avisos do pandas
df_100 = df_100.sort_values(by='sucesso', ascending=True)

if not df_100.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    barras = ax.barh(df_100['teste'], df_100['sucesso'], color='#4e79a7')

    ax.set_title('Taxa de Sucesso por Cenário (100 Repetições)', fontsize=14)
    ax.set_xlabel('Taxa de Sucesso (0.0 a 1.0)')
    ax.set_xlim(0, 1.15) # Aumentei um pouco para caber o texto

    # Rotular barras
    for bar in barras:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{int(width*100)}%', va='center', fontweight='bold')

    plt.tight_layout()
    try:
        plt.savefig('grafico_geral_100reps.png', dpi=300)
    except:
        plt.savefig('grafico_geral_100reps.png', dpi=300) # Tenta na raiz se a pasta img não existir

# ATENÇÃO: Os nomes aqui devem ser IDÊNTICOS aos que apareceram no print "Nomes encontrados"
top3 = ['Elitismo', 'Recomb_2Filhos', 'BestOfN']

# Filtra o DataFrame
df_top3 = df[df['teste'].isin(top3)]

# Se o filtro retornar vazio, avisa o usuário
if df_top3.empty:
    print("\nERRO CRÍTICO NO GRÁFICO 2: Nenhum dos nomes do top3 foi encontrado no CSV.")
    print(f"O código procurou por: {top3}")
    print(f"Mas no CSV tem: {df['teste'].unique()}")
    print("DICA: Verifique se no CSV os nomes estão escritos exatamente iguais (Maiúsculas/Minúsculas).")
else:
    # Separa os dados com segurança
    dados_100 = df_top3[df_top3['repeticoes'] == 100].set_index('teste').reindex(top3)
    dados_1000 = df_top3[df_top3['repeticoes'] == 1000].set_index('teste').reindex(top3)

    # Prepara o plot
    x = np.arange(len(top3))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotar as barras
    rects1 = ax.bar(x - width/2, dados_100['geracaoMedia'], width, label='100 Reps', color='#e15759')
    rects2 = ax.bar(x + width/2, dados_1000['geracaoMedia'], width, label='1000 Reps (Otimizado)', color='#59a14f')

    ax.set_title('Comparativo de Gerações Médias: Otimização Drástica', fontsize=14)
    ax.set_ylabel('Número de Gerações (Menor é melhor)')
    ax.set_xticks(x)
    ax.set_xticklabels(top3)
    ax.legend()

    # Adicionar valores em cima das barras (trata erro se tiver valor vazio)
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')

    plt.tight_layout()
    try:
        plt.savefig('img/grafico_comparativo.png', dpi=300)
        print("Gráfico 2 salvo com sucesso!")
    except:
        plt.savefig('img/grafico_comparativo.png', dpi=300)