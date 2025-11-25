import random
import time
from typing import List, Tuple, Optional

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

# ---------- Seleção proporcional (roleta) ----------
def roulette_wheel_selection(population: List[List[int]], fit_values: List[int]) -> List[List[int]]:
    """
    Retorna uma lista de pais escolhidos (pares consecutivos formam pares de pais).
    Aqui apenas devolvemos o conjunto de candidatos (pop_size) por conveniência; 
    a recombinação irá retirar 2 por 2 para gerar filhos.
    """
    # transform fitness (menor é melhor) -> aptidão maior = melhor
    # apt = max_attacks - attacks + epsilon
    max_attacks = max(fit_values)
    # convert attacks -> aptitudes (maior apt -> mais chances)
    aptitudes = [(max_attacks - f) + 1e-6 for f in fit_values]  # +epsilon para evitar zero
    total = sum(aptitudes)
    probs = [a/total for a in aptitudes]
    cum = []
    s = 0.0
    for p in probs:
        s += p
        cum.append(s)
    selected = []
    for _ in range(len(population)):
        r = random.random()
        # find index
        lo, hi = 0, len(cum)-1
        while lo < hi:
            mid = (lo+hi)//2
            if r <= cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        selected.append(population[lo][:])  # copy
    return selected

# ---------- Cruzamento: corte único com reparo por ordem (gera 1 filho) ----------
def one_point_order_crossover(parent1: List[int], parent2: List[int], cut: Optional[int] = None) -> List[int]:
    """
    Faz um corte em 'cut' (entre 1 e n-1). Filho recebe prefixo do parent1 (0..cut-1)
    e depois os genes faltantes na ordem em que aparecem no parent2.
    Esse método garante permutação válida e mantém a ideia de 'um ponto de corte'.
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

# ---------- Mutação: swap ----------
def swap_mutation(individual: List[int], mutation_rate: float = 0.10) -> List[int]:
    """
    Aplica troca de duas posições com probabilidade mutation_rate ao indivíduo.
    """
    ind = individual[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
    return ind

# ---------- Próxima geração: apenas filhos (padrão) ----------
# (já implementado ao gerar exatamente pop_size filhos)

# ---------- Algoritmo Genético (versão base) ----------
def genetic_algorithm_base(
    n: int = 8,
    pop_size: int = 100,
    mixing_number: int = 2,    # rho = 2 (assumido)
    selection_method: str = "proportional",  # apenas proporcional aqui (padrão da aula)
    crossover_fn = one_point_order_crossover,
    mutation_rate: float = 0.10,
    max_generations: int = 1000,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Tuple[Optional[List[int]], int, float]:
    """
    Retorna (solução ou None, gerações usadas, tempo_em_segundos).
    Comportamento padrão reproduzindo a versão da aula.
    """
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    population = init_population(pop_size, n)
    gen = 0

    # avaliar
    fit_values = [fitness(ind) for ind in population]
    if verbose:
        print(f"[G{gen}] best fitness: {min(fit_values)}")

    while gen < max_generations:
        gen += 1
        # seleção
        if selection_method == "proportional":
            parents_pool = roulette_wheel_selection(population, fit_values)
        else:
            raise ValueError("selection_method desconhecido. Somente 'proportional' implementado no base.")
        # recombinar: gerar pop_size filhos (2 filhos por par, usando 2 pais)
        children = []
        # embaralha a pool para emparelhar aleatoriamente
        random.shuffle(parents_pool)
        # emparelhar consecutivamente e gerar 2 filhos por casal
        for i in range(0, len(parents_pool), 2):
            p1 = parents_pool[i]
            p2 = parents_pool[i+1 if i+1 < len(parents_pool) else 0]
            # gerar 2 filhos usando corte único com reparo (um de cada direção)
            child1 = crossover_fn(p1, p2)
            child2 = crossover_fn(p2, p1)
            # mutação
            child1 = swap_mutation(child1, mutation_rate)
            child2 = swap_mutation(child2, mutation_rate)
            children.append(child1)
            children.append(child2)
        population = children
        fit_values = [fitness(ind) for ind in population]

        best_fit = min(fit_values)
        if verbose and gen % 50 == 0:
            print(f"[G{gen}] best fitness: {best_fit}")
        # checar solução
        if 0 in fit_values:
            idx = fit_values.index(0)
            elapsed = time.time() - start_time
            if verbose:
                print(f"Solução encontrada na geração {gen} em {elapsed:.4f}s")
            return population[idx], gen, elapsed

    elapsed = time.time() - start_time
    # não encontrou solução
    if verbose:
        print(f"Nenhuma solução encontrada em {max_generations} gerações (melhor fitness {min(fit_values)})")
    return None, gen, elapsed

# ---------- Exemplo de uso ----------
if __name__ == "__main__":
    # executa 1 vez para demonstrar
    sol, gens, t = genetic_algorithm_base(verbose=True, seed=42)
    if sol:
        print("Solução:", sol, "Gerações:", gens, "Tempo(s):", round(t,4))
    else:
        print("Sem solução. Gerações:", gens, "Tempo(s):", round(t,4))

    # exemplo: rodar 10 repetições rápidas para ver taxa de sucesso
    reps = 10
    successes = 0
    total_gens = 0
    total_time = 0.0
    for s in range(reps):
        sol, gens, t = genetic_algorithm_base(seed=s)
        if sol:
            successes += 1
            total_gens += gens
            total_time += t
    print(f"\nEm {reps} repetições: sucessos = {successes}, taxa = {successes/reps:.2f}")
    if successes:
        print(f"generaçõs média (só sucessos) = {total_gens/successes:.2f}, tempo médio = {total_time/successes:.4f}s")
