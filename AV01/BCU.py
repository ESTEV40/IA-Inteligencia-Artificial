import heapq
import itertools

"""
Esse é um No do grafo de busca. Não a um no do Mapa do problema.
Representa caminhos até o determinado No.
Ex:
A -> B -> C
"""
class No:
    def __init__(self, estadoAtual,pai=None, custo = 0) :
        self.estadoAtual = estadoAtual
        self.pai = pai
        self.custo = custo
    
"""
Esse é o Objeto do Problema, onde têm o grafo do mapa com 
"""
class mapProblem:
    def __init__(self,estadoInicial,estadoFinal,grafo) :
        self.estadoInicial = estadoInicial
        self.estadoFinal = estadoFinal
        self.grafo = grafo
    
    def isObjetivo(self, estado):
        return estado == self.estadoFinal
    
    def expand(self, noPai) :
        filhos = []

        estadoPai = noPai.estadoAtual

        for(filho, custoPasso) in self.grafo.get(estadoPai,[]) :
            custoPassoNovo = noPai.custo + custoPasso

            noFilho = No(
                estadoAtual = filho,
                pai = noPai,
                custo = custoPassoNovo
            )

            filhos.append(noFilho)
        
        return filhos

"""
Esta função define a 'prioridade' de um nó na fronteira. Para a BCU,
a prioridade é simplesmente o custo total do caminho (g(n)) acumulado
desde o estado inicial até o nó atual.
"""
def f(no) :
    return no.custo

"""
Esta função utiliza uma fila de prioridade ('fronteira') para sempre
expandir o nó com o menor custo total acumulado desde o início.
Um dicionário ('visitados') é usado para registrar o melhor caminho
encontrado até cada estado, evitando ciclos e a exploração de
caminhos mais caros que um já conhecido.
"""
def BCU(problema, f) :

    # Cria o nó inicial da busca
    noInicial = No(estadoAtual=problema.estadoInicial)

    # Inicializa a fronteira e o contador de desempate
    contador = itertools.count()
    fronteira = []
    heapq.heappush(fronteira, (f(noInicial), next(contador), noInicial))
    
    # Inicializa o dicionário de visitados com o melhor caminho até o início
    visitados = {problema.estadoInicial : noInicial}

    # Loop principal: continua enquanto houver nós na fronteira
    while fronteira:
        # Remove o nó de menor custo da fronteira
        _prioridade, _contador, noAtual = heapq.heappop(fronteira)

        # Verifica se o nó atual é o objetivo
        if problema.isObjetivo(noAtual.estadoAtual) :
            return noAtual
        
        # Gera os filhos do nó atual
        for filho in problema.expand(noAtual) :
            estadoFilho = filho.estadoAtual
            # Verifica se o filho é um estado novo ou um caminho mais barato
            if estadoFilho not in visitados or filho.custo < visitados[estadoFilho].custo :
                # Atualiza o dicionário com o novo caminho, que é melhor
                visitados[estadoFilho] = filho
                # Adiciona o filho à fronteira para ser explorado
                heapq.heappush(fronteira, (f(filho), next(contador), filho))
    
    # Retorna None se a fronteira esvaziar e o objetivo não for encontrado
    return None

"""
Essa função constroi uma lista dos estados visitados do resultado em ordem inversa
"""
def constroiCaminho(no) :
    # Inicializa as variáveis para o caminho e custo
    caminho = []
    custoTotal = no.custo
    temp = no

    # Percorre a trilha de pais do nó final até o inicial
    while temp is not None:
        caminho.append(temp.estadoAtual)
        temp = temp.pai
    # Inverte a lista para ter o caminho na ordem correta
    return caminho[::-1], custoTotal

mapa = {
    'Arad': [('Zerind', 75), ('Sibiu', 140), ('Timisoara', 118)],
    'Zerind': [('Arad', 75), ('Oradea', 71)],
    'Oradea': [('Zerind', 71), ('Sibiu', 151)],
    'Sibiu': [('Arad', 140), ('Oradea', 151), ('Fagaras', 99), ('Rimnicu Vilcea', 80)],
    'Timisoara': [('Arad', 118), ('Lugoj', 111)],
    'Lugoj': [('Timisoara', 111), ('Mehadia', 70)],
    'Mehadia': [('Lugoj', 70), ('Drobeta', 75)],
    'Drobeta': [('Mehadia', 75), ('Craiova', 120)],
    'Craiova': [('Drobeta', 120), ('Rimnicu Vilcea', 146), ('Pitesti', 138)],
    'Rimnicu Vilcea': [('Sibiu', 80), ('Craiova', 146), ('Pitesti', 97)],
    'Fagaras': [('Sibiu', 99), ('Bucharest', 211)],
    'Pitesti': [('Rimnicu Vilcea', 97), ('Craiova', 138), ('Bucharest', 101)],
    'Giurgiu': [('Bucharest', 90)],
    'Bucharest': [('Fagaras', 211), ('Pitesti', 101), ('Giurgiu', 90)]
}

def main() :
    # Cria a instância do problema
    problema = mapProblem('Arad','Bucharest',mapa)

    # Executa a busca
    buscaResultado = BCU(problema, f)

    # Verifica se a busca encontrou uma solução
    if buscaResultado:
        caminho, custo = constroiCaminho(buscaResultado)
        print(f"Caminho: {caminho}")
        print(f"Custo Total: {custo}")
    else:
        print("Nenhum caminho encontrado.")


if __name__ == "__main__" :
    main()