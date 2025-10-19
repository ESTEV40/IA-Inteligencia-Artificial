import math

"""
Esse é um No do grafo de busca. Não a um no do Mapa do problema.
Representa caminhos até o determinado No.
Ex:
A -> B -> C
"""
class No:
    def __init__(self, estadoAtual,pai=None, custo = 0, fCusto = 0) :
        self.estadoAtual = estadoAtual
        self.pai = pai
        self.custo = custo
        self.fCusto = fCusto # custo + heuristica[estadoAtual]
    
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
f(no) é a função de prioridade da fronteira
"""
def f(no) :
    return no.custo + heuristica_bucareste.get(no.estadoAtual, math.inf)

"""
RBFS() é a função inicial que antecede a função recursiva em si, inicializa os parâmetros de busca
é o Caso Base.
"""
def RBFS(problema) :

    #Inicizalização
    noInicial = No(estadoAtual = problema.estadoInicial)
    noInicial.fCusto = f(noInicial)
    
    #Primeira chamada Recursiva
    resultado, _fCusto = search(problema, noInicial, f, fLimit = math.inf)
    
    #Formatação do Resultado
    if resultado :
        caminho = []
        custoTotal = resultado.custo
        temp = resultado

        while temp is not None:
            caminho.append(temp.estadoAtual)
            temp = temp.pai
        return caminho[::-1], custoTotal
    else :
        return None, math.inf

"""
search() é a função recursiva em si
"""
def search(problema, noAtual, f, fLimit) :
    
    #Verifica se o Nó Atual é o Objetivo
    if problema.isObjetivo(noAtual.estadoAtual):
        return noAtual, None
    
    #Retorna os Filhos do Nó Atual, caso seja nó sem saída, retorna infinito
    filhos = problema.expand(noAtual)
    if not filhos :
        return None, math.inf
    
    #Atualiza os valores da função(custo) para todos os filhos
    for filho in filhos :
        filho.fCusto = max(f(filho), noAtual.fCusto)

    while True:
        #Ordena os filhos de acordo com o custo
        filhos.sort(key = lambda no : no.fCusto)
        melhorFilho = filhos[0]

        #Se o custo do caminho aumentar
        #Ele retorna None para o Resultado e o Custo do caminho para atualizar
        if melhorFilho.fCusto > fLimit :
            return None, melhorFilho.fCusto
        
        #Backup do segundo melhor filho para lembrar caso o custo do caminho aumente muito
        custoSegundoFilhoBackup = filhos[1].fCusto if len(filhos) > 1 else math.inf

        #Chamada Recursiva
        resultado, fCustoAtualizado = search(problema, melhorFilho, f, min(fLimit,custoSegundoFilhoBackup))

        #Quando chegar ao objetivo, ele retorna o resultado
        if resultado is not None:
            return resultado, None

        #Atualiza o custo caso ele seja maior que o limite
        melhorFilho.fCusto = fCustoAtualizado


mapaArad = {
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
heuristica_bucareste = {
    'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Drobeta': 242,
    'Eforie': 161, 'Fagaras': 176, 'Giurgiu': 77, 'Hirsova': 151,
    'Iasi': 226, 'Lugoj': 244, 'Mehadia': 241, 'Neamt': 234,
    'Oradea': 380, 'Pitesti': 100, 'Rimnicu Vilcea': 193, 'Sibiu': 253,
    'Timisoara': 329, 'Urziceni': 80, 'Vaslui': 199, 'Zerind': 374
}

def main() :
    problema = mapProblem('Arad','Bucharest',mapaArad)
    resultado, custo = RBFS(problema)

    # Verifica se a busca encontrou uma solução
    if resultado:
        print(f"Caminho: {resultado}")
        print(f"Custo Total: {custo}")
    else:
        print("Nenhum caminho encontrado.")

if __name__ == "__main__" :
    main()