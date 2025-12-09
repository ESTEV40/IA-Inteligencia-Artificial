import Ag8rainhasExtend as ag

# Função auxiliar para não repetir código
def rodar_cenario(nome_cenario, config_dict, repeticoes=1000):
    print(f"\n{'='*60}")
    print(f"INICIANDO: {nome_cenario}")
    print(f"Configuração: {config_dict}")
    print(f"{'='*60}")
    
    sucessos = 0
    soma_gens = 0
    soma_tempo = 0
    
    # Loop de 1000 repetições
    for i in range(repeticoes):
        sol, gens, t = ag.genetic_algorithm_extended(seed=i, verbose=False, **config_dict)
        
        if sol:
            sucessos += 1
            soma_gens += gens
            soma_tempo += t
            
        if (i+1) % 100 == 0:
            print(f"Processando {i+1}/{repeticoes}...")

    # Resultados
    media_gens = (soma_gens / sucessos) if sucessos > 0 else 0
    media_tempo = (soma_tempo / sucessos) if sucessos > 0 else 0
    taxa = sucessos / repeticoes

    print(f"\n[RESULTADO FINAL - {nome_cenario}]")
    print(f"Taxa de Sucesso: {taxa:.2f} ({taxa*100}%)")
    print(f"Gerações Média:  {media_gens:.2f}")
    print(f"Tempo Médio:     {media_tempo:.4f}s")
    print("-" * 60)

def main() : 
    # CENÁRIO 1: ELITISMO
    # Muda apenas a estratégia de próxima geração. O resto pega do padrão.
    config_elitismo = {
        "next_generation_strategy": "elitism" 
    }

    # CENÁRIO 2: RECOMBINAÇÃO (2 FILHOS + CROSSOVER BIDIRECIONAL)
    config_2filhos = {
        "crossover_fn": ag.one_point_order_crossover_bidirectional,
        "offspring_count": 2,
        "next_generation_strategy": "descendants" # Reforçando que não é elitismo
    }

    # CENÁRIO 3: SELEÇÃO BEST-OF-N (TORNEIO)
    # Muda o método de seleção.
    config_best_of_n = {
        "selection_method": "best_of_n",
        "selection_pool_size": 5,
        "next_generation_strategy": "descendants" # Reforçando que não é elitismo
    }

    # --- EXECUÇÃO ---
    rodar_cenario("Elitismo", config_elitismo)
    rodar_cenario("Recombinação 2 Filhos", config_2filhos)
    rodar_cenario("Seleção Best-of-N", config_best_of_n)

if __name__ == "__main__":
    main()