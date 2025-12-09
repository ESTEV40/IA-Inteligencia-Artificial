import random
import time
from typing import List, Tuple, Optional, Callable

# ---------- Helpers / Fitness ----------
def fitness(individual: List[int]) -> int:
    """
    Número de pares de rainhas que se atacam (menor é melhor).
    Representação: individual[i] = linha (1..8) da rainha na coluna i.
    """
    n = len(individual)
    attacks = 0
    for i in range(n):
        for j in range(i+1, n):
            # mesma linha
            if individual[i] == individual[j]:
                attacks += 1
            # mesma diagonal
            if abs(individual[i] - individual[j]) == abs(i - j):
                attacks += 1
    return attacks

def is_solution(ind: List[int]) -> bool:
    return fitness(ind) == 0

# ---------- Inicialização ----------
def random_individual(n: int = 8) -> List[int]:
    """Permutação das linhas 1..n"""
    perm = list(range(1, n+1))
    random.shuffle(perm)
    return perm

def init_population(pop_size: int = 100, n: int = 8) -> List[List[int]]:
    return [random_individual(n) for _ in range(pop_size)]

# ---------- Seleção Proporcional (Roleta) ----------
def roulette_wheel_selection(population: List[List[int]], fit_values: List[int]) -> List[List[int]]:
    """
    Retorna uma lista de indivíduos selecionados com probabilidade proporcional à aptidão.
    """
    # Transformar fitness (menor é melhor) -> aptidão maior = melhor
    max_attacks = max(fit_values)
    aptitudes = [(max_attacks - f) + 1e-6 for f in fit_values]  # +epsilon para evitar zero
    total = sum(aptitudes)
    probs = [a/total for a in aptitudes]
    cum = []
    s = 0.0
    for p in probs:
        s += p
        cum.append(s)
        
    selected = []
    pop_size = len(population)
    
    # Se ρ=2, precisamos de um número par de pais para emparelhar. Usamos pop_size como pool.
    # Se ρ=1, precisamos de pop_size pais.
    # Para simplificar, o tamanho da pool de pais é pop_size.
    pool_size = pop_size 
    
    for _ in range(pool_size):
        r = random.random()
        lo, hi = 0, len(cum)-1
        # Busca binária
        while lo < hi:
            mid = (lo+hi)//2
            if r <= cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        selected.append(population[lo][:])  # Retorna cópia
    return selected

# ---------- Seleção por Torneio Determinístico (Melhores de N) ----------
def best_of_n_selection(
    population: List[List[int]], 
    fit_values: List[int], 
    pop_size: int, 
    mixing_number: int, 
    selection_pool_size: int
) -> List[List[int]]:
    """
    Seleciona 'mixing_number' pais escolhendo os 'mixing_number' mais aptos
    de 'selection_pool_size' indivíduos selecionados aleatoriamente.
    Repete até ter 'pop_size' indivíduos para a pool de pais (não pares).
    """
    parents_pool = []
    n_individuals = len(population)
    
    # Quantos conjuntos de pais precisamos? 
    # Para manter o tamanho da próxima geração em pop_size, precisamos de pop_size pais 
    # se mixing_number=1, ou pop_size pais (emparelhados 2 a 2) se mixing_number=2.
    # Para ser geral, criamos uma pool do tamanho da população (pop_size) e depois
    # a função de recombinação a usará em grupos de 'mixing_number'.
    
    while len(parents_pool) < pop_size:
        # 1. Seleciona aleatoriamente 'selection_pool_size' indivíduos (com reposição)
        indices = [random.choice(range(n_individuals)) for _ in range(selection_pool_size)]
        
        # 2. Reúne indivíduos e seus fitness
        candidates = [(population[i], fit_values[i]) for i in indices]
        
        # 3. Ordena pela aptidão (fitness, menor é melhor)
        # candidates.sort(key=lambda x: x[1]) # Já fazemos isso com a aptidão.
        
        # 4. Seleciona os 'mixing_number' mais aptos
        # Para otimizar, calculamos a aptidão só aqui se necessário, mas fitness já serve para ordem.
        
        # Ordena candidatos pelo fitness (menor é melhor)
        candidates.sort(key=lambda x: x[1]) 
        
        # Adiciona os 'mixing_number' melhores. 
        # CUIDADO: Se selection_pool_size < mixing_number, pega todos.
        
        num_to_select = min(mixing_number, selection_pool_size)
        
        # Adiciona os indivíduos (índice 0 da tupla) à pool de pais
        for i in range(num_to_select):
            parents_pool.append(candidates[i][0][:]) # Cópia do indivíduo
            
    # Trunca para pop_size (para manter o tamanho)
    return parents_pool[:pop_size]

# ---------- Cruzamento: corte único com reparo por ordem (gera 1 filho) ----------
def one_point_order_crossover(parent1: List[int], parent2: List[int], cut: Optional[int] = None) -> List[int]:
    """
    Faz um corte em 'cut'. Filho recebe prefixo do parent1 e genes faltantes do parent2 na ordem.
    Garante permutação válida.
    """
    n = len(parent1)
    if cut is None:
        cut = random.randint(1, n-1)
    child = parent1[:cut]
    # preencher com elementos de parent2 que ainda não estão no child, mantendo a ordem
    for gene in parent2:
        if gene not in child:
            child.append(gene)
    return child

# ---------- Cruzamento: corte único que gera 2 filhos (bidirecional) ----------
# Adaptado do original: gera um filho P1-P2 e outro P2-P1
def one_point_order_crossover_bidirectional(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Gera dois filhos: (P1 prefixo + P2 restante), (P2 prefixo + P1 restante).
    """
    n = len(parent1)
    cut = random.randint(1, n-1)
    
    # Filho 1: P1 prefixo, P2 restante
    child1 = parent1[:cut]
    for gene in parent2:
        if gene not in child1:
            child1.append(gene)
            
    # Filho 2: P2 prefixo, P1 restante
    child2 = parent2[:cut]
    for gene in parent1:
        if gene not in child2:
            child2.append(gene)
            
    return child1, child2

# ---------- Mutação: swap ----------
def swap_mutation(individual: List[int], mutation_rate: float = 0.10) -> List[int]:
    """
    Aplica troca de duas posições com probabilidade mutation_rate ao indivíduo.
    """
    ind = individual[:]
    if random.random() < mutation_rate:
        n = len(ind)
        if n >= 2:
            i, j = random.sample(range(n), 2)
            ind[i], ind[j] = ind[j], ind[i]
    return ind

# ---------- Algoritmo Genético Estendido ----------
def genetic_algorithm_extended(
    n: int = 8,
    pop_size: int = 100,
    mixing_number: int = 2,    # ρ = 2 (default)
    selection_method: str = "proportional",  # 'proportional' ou 'best_of_n'
    selection_pool_size: int = 5, # N para seleção 'best_of_n' (só usado se selection_method='best_of_n')
    crossover_fn: Callable = one_point_order_crossover,
    offspring_count: int = 1, # Número de filhos gerados por 'mixing_number' pais (1 ou 2 para ρ=2)
    mutation_rate: float = 0.10,
    next_generation_strategy: str = "descendants", # 'descendants' ou 'elitism'
    max_generations: int = 1000,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Tuple[Optional[List[int]], int, float]:
    """
    Retorna (solução ou None, gerações usadas, tempo_em_segundos).
    Comportamento padrão reproduz a versão da aula.
    """
    
    # --- Validação de Parâmetros ---
    if selection_method not in ["proportional", "best_of_n"]:
        raise ValueError("selection_method deve ser 'proportional' ou 'best_of_n'.")
    if next_generation_strategy not in ["descendants", "elitism"]:
        raise ValueError("next_generation_strategy deve ser 'descendants' ou 'elitism'.")
        
    # As funções de cruzamento bidirecional geram 2 filhos por casal (rho=2)
    if crossover_fn == one_point_order_crossover_bidirectional:
        offspring_count = 2 
    
    if mixing_number == 1 and selection_method == "best_of_n":
        print("Aviso: ρ=1 com 'best_of_n' usa o 1 melhor de N (cria uma pool de 'pop_size' pais).")

    if mixing_number != 2 and crossover_fn != one_point_order_crossover:
        # Para simplificar, só usamos o crossover de 1 filho com rho != 2.
        # Implementar crossover geral para rho>2 é complexo e foge do escopo do exerc.
        crossover_fn = one_point_order_crossover
        if verbose:
             print("Aviso: Para ρ!=2, a função de recombinação foi ajustada para 'one_point_order_crossover' (gera 1 filho).")
        offspring_count = 1 

    # --- Inicialização ---
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    
    population = init_population(pop_size, n)
    gen = 0
    
    # Avaliar
    fit_values = [fitness(ind) for ind in population]
    
    best_ind_gen0 = None
    if next_generation_strategy == "elitism":
        best_fit_gen0 = min(fit_values)
        idx_best_gen0 = fit_values.index(best_fit_gen0)
        best_ind_gen0 = population[idx_best_gen0][:] # Cópia do melhor
    
    if verbose:
        print(f"[G{gen}] best fitness: {min(fit_values)}")

    # Checar solução inicial
    if 0 in fit_values:
        idx = fit_values.index(0)
        elapsed = time.time() - start_time
        if verbose:
            print(f"Solução encontrada na geração {gen} em {elapsed:.4f}s (População Inicial)")
        return population[idx], gen, elapsed
        
    # --- Loop do AG ---
    while gen < max_generations:
        gen += 1
        
        # 1. Seleção
        if selection_method == "proportional":
            parents_pool = roulette_wheel_selection(population, fit_values)
        elif selection_method == "best_of_n":
            parents_pool = best_of_n_selection(
                population, fit_values, pop_size, mixing_number, selection_pool_size
            )
        
        # 2. Recombinar: gerar ~pop_size filhos (ou pop_size, se possível)
        children = []
        random.shuffle(parents_pool) # Emparelhar aleatoriamente
        
        # O loop deve percorrer a pool de pais em grupos de 'mixing_number'
        step = mixing_number
        for i in range(0, len(parents_pool), step):
            # Lida com o final da pool caso o tamanho não seja múltiplo de 'mixing_number'
            parents = parents_pool[i:i+step] 
            
            # Se não houver 'mixing_number' pais suficientes (ex.: pool_size 100, rho 3, ultimo grupo de 1)
            if len(parents) < mixing_number:
                # Simplificação: recicla pais da pool para fechar o grupo de 'mixing_number'
                # Ou apenas usa o que resta e gera filhos insuficientes (opção 2). Faremos a opção 2.
                # A próxima geração terá < pop_size se o número de filhos for 1 e o grupo final for incompleto.
                continue 
                
            p1 = parents[0]
            p2 = parents[1] if mixing_number >= 2 else None
            
            # Geração de filhos
            if mixing_number == 1:
                # Reprodução Assexuada (equivalente a busca estocástica de feixe)
                child = p1[:] # Cópia
                # Mutação (crossover não se aplica aqui, a mutação gera variação)
                child = swap_mutation(child, mutation_rate)
                children.append(child)
            
            elif mixing_number == 2:
                # Recombinação com 2 pais
                if crossover_fn == one_point_order_crossover_bidirectional:
                    # Gera 2 filhos
                    child1, child2 = crossover_fn(p1, p2)
                    child1 = swap_mutation(child1, mutation_rate)
                    child2 = swap_mutation(child2, mutation_rate)
                    children.append(child1)
                    children.append(child2)
                    
                else: # crossover_fn == one_point_order_crossover (default)
                    # Gera 1 ou 2 filhos dependendo de offspring_count
                    child1 = crossover_fn(p1, p2)
                    child1 = swap_mutation(child1, mutation_rate)
                    children.append(child1)
                    # Sempre gerar o segundo filho para manter pop_size
                    # (mesmo que offspring_count=1, precisamos de pop_size filhos no total)
                    child2 = crossover_fn(p2, p1) # Inverte pais
                    child2 = swap_mutation(child2, mutation_rate)
                    children.append(child2)
                        
            elif mixing_number >= 3:
                # Simplificação: Usa o primeiro pai como base, os demais para preencher
                child = p1[:]
                for parent_idx in range(1, mixing_number):
                     # Tenta inserir genes dos outros pais
                     for gene in parents[parent_idx]:
                        if gene not in child:
                            child.append(gene)
                            if len(child) == n: break # Cheio
                     if len(child) == n: break
                
                # O restante é preenchido de forma aleatória se ainda não estiver completo (não deveria ocorrer)
                if len(child) < n:
                    missing_genes = [g for g in range(1, n+1) if g not in child]
                    random.shuffle(missing_genes)
                    child.extend(missing_genes)
                    
                child = swap_mutation(child, mutation_rate)
                children.append(child)
                
            # Trunca filhos para manter o pop_size (pode ocorrer se offspring_count=2 e pop_size for ímpar)
            # Para manter o AG, é melhor não truncar e deixar a população variar levemente no tamanho, 
            # ou garantir que o loop de pais gere o número exato de filhos.
            
        # 3. Montagem da Próxima Geração
        
        # 3a. Elitismo: o melhor da geração anterior
        if next_generation_strategy == "elitism":
            # Obtém o melhor da geração atual (population) antes de ser substituída
            current_best_fit = min(fit_values)
            idx_current_best = fit_values.index(current_best_fit)
            current_best_ind = population[idx_current_best][:]

            # 3b. Escolhe os melhores filhos (se houver elitismo, um slot é para o melhor pai)
            num_descendants = pop_size
            if len(children) > 0 and pop_size > 0 and current_best_fit < min([fitness(c) for c in children]): # Se o melhor pai é melhor que o melhor filho
                 num_descendants = pop_size - 1
                 children.sort(key=fitness) # Ordena filhos
                 population = [current_best_ind] + children[:num_descendants] # O melhor pai e os (N-1) melhores filhos
            else:
                 if len(children) > 0:
                     children.sort(key=fitness)
                     population = children[:pop_size] # Os N melhores filhos
                 else:
                     # Se não há filhos, mantém a população atual
                     population = population[:pop_size]

        else: # next_generation_strategy == "descendants" (versão original)
            if len(children) >= pop_size:
                population = children[:pop_size] # Apenas os filhos, truncados
            else:
                # Se não há filhos suficientes, completa com a população anterior
                population = children + population[:pop_size - len(children)]

        # 4. Avaliar nova população
        fit_values = [fitness(ind) for ind in population]
        best_fit = min(fit_values)

        if verbose and gen % 50 == 0:
            print(f"[G{gen}] best fitness: {best_fit}")
        
        # 5. Checar solução
        if 0 in fit_values:
            idx = fit_values.index(0)
            elapsed = time.time() - start_time
            if verbose:
                print(f"Solução encontrada na geração {gen} em {elapsed:.4f}s")
            return population[idx], gen, elapsed

    elapsed = time.time() - start_time
    # Não encontrou solução
    if verbose:
        print(f"Nenhuma solução encontrada em {max_generations} gerações (melhor fitness {min(fit_values)})")
    return None, gen, elapsed

# ---------- Testes para o Exercício ----------
def run_experiments(reps: int = 100):
    """Executa o algoritmo com diferentes configurações para o exercício."""
    
    # --- 1. Valores Padrão (Reproduzindo o comportamento original) ---
    print("## 1. Configuração Padrão (Original)")
    print(f"   (pop_size=100, ρ=2, selection='proportional', crossover='1-point-1-child', mutation=10%, next_gen='descendants')")
    results = [genetic_algorithm_extended(verbose=False, seed=s, pop_size=100, mixing_number=2, selection_method="proportional", crossover_fn=one_point_order_crossover, offspring_count=1, mutation_rate=0.10, next_generation_strategy="descendants", max_generations=1000) for s in range(reps)]
    report_results(results, reps)
    
    # --- 2. Alteração de Parâmetros ---
    
    # 2.1. Tamanho da População (Exemplo: 50 indivíduos)
    print("\n## 2.1. Teste: Tamanho da População (pop_size=50)")
    results_pop50 = [genetic_algorithm_extended(verbose=False, seed=s, pop_size=50, mixing_number=2, selection_method="proportional", crossover_fn=one_point_order_crossover, offspring_count=1, mutation_rate=0.10, next_generation_strategy="descendants") for s in range(reps)]
    report_results(results_pop50, reps)
    
    # 2.2. Número de Mistura (ρ = 1) - Busca Estocástica de Feixe
    # Requer offspring_count=1 e crossover_fn ignorado
    print("\n## 2.2. Teste: Número de Mistura (ρ=1, Estocástica de Feixe)")
    results_rho1 = [genetic_algorithm_extended(verbose=False, seed=s, pop_size=100, mixing_number=1, selection_method="proportional", crossover_fn=one_point_order_crossover, offspring_count=1, mutation_rate=0.10, next_generation_strategy="descendants") for s in range(reps)]
    report_results(results_rho1, reps)

    # 2.3. Estratégias de Seleção (Exemplo: Seleção 'Best-of-N', N=5, ρ=2)
    print("\n## 2.3. Teste: Seleção 'Best-of-N' (selection='best_of_n', N=5, ρ=2)")
    results_bestofn = [genetic_algorithm_extended(verbose=False, seed=s, pop_size=100, mixing_number=2, selection_method="best_of_n", selection_pool_size=5, crossover_fn=one_point_order_crossover, offspring_count=1, mutation_rate=0.10, next_generation_strategy="descendants") for s in range(reps)]
    report_results(results_bestofn, reps)

    # 2.4. Procedimentos de Recombinação (Exemplo: 1-ponto, 2 filhos)
    print("\n## 2.4. Teste: Recombinação Bidirecional (1-ponto, 2 filhos/casal, ρ=2)")
    results_bidir = [genetic_algorithm_extended(verbose=False, seed=s, pop_size=100, mixing_number=2, selection_method="proportional", crossover_fn=one_point_order_crossover_bidirectional, offspring_count=2, mutation_rate=0.10, next_generation_strategy="descendants") for s in range(reps)]
    report_results(results_bidir, reps)
    
    # 2.5. Taxa de Mutação (Exemplo: 5%)
    print("\n## 2.5. Teste: Taxa de Mutação (mutation_rate=5%)")
    results_mut05 = [genetic_algorithm_extended(verbose=False, seed=s, pop_size=100, mixing_number=2, selection_method="proportional", crossover_fn=one_point_order_crossover, offspring_count=1, mutation_rate=0.05, next_generation_strategy="descendants") for s in range(reps)]
    report_results(results_mut05, reps)
    
    # 2.6. Montagem da Próxima Geração (Exemplo: Elitismo)
    print("\n## 2.6. Teste: Montagem da Próxima Geração (next_gen='elitism')")
    results_elitism = [genetic_algorithm_extended(verbose=False, seed=s, pop_size=100, mixing_number=2, selection_method="proportional", crossover_fn=one_point_order_crossover, offspring_count=1, mutation_rate=0.10, next_generation_strategy="elitism") for s in range(reps)]
    report_results(results_elitism, reps)
    

def report_results(results: List[Tuple[Optional[List[int]], int, float]], reps: int):
    """Função auxiliar para relatar os resultados de uma bateria de testes."""
    successes = sum(1 for sol, _, _ in results if sol is not None)
    total_gens = sum(gens for sol, gens, _ in results if sol is not None)
    total_time = sum(t for sol, _, t in results if sol is not None)
    
    success_rate = successes / reps
    
    print(f"   -> Repetições: {reps}, Sucessos: {successes}, **Taxa de Sucesso: {success_rate:.2f}**")
    if successes > 0:
        print(f"   -> Gerações Média (só sucessos): {total_gens / successes:.2f}")
        print(f"   -> Tempo Médio (só sucessos): {total_time / successes:.4f}s")
    
# ---------- Exemplo de uso (Rodar os testes) ----------
if __name__ == "__main__":
    # Teste para o comportamento padrão (similar ao arquivo original)
    print("--- Execução de Teste Padrão (seed=42) ---")
    sol, gens, t = genetic_algorithm_extended(verbose=True, seed=42)
    if sol:
        print("Solução:", sol, "Gerações:", gens, "Tempo(s):", round(t,4))
    else:
        print("Sem solução. Gerações:", gens, "Tempo(s):", round(t,4))
    print("-------------------------------------------\n")

    # Rodar as 100 repetições para o exercício
    print("--- Resultados de 100 Repetições com Variações de Parâmetros ---")
    run_experiments(reps=100)