import random
import numpy as np
from abc import abstractmethod

class Estado:
    def __init__(self, estado):
        self.estado = estado

class Acao:
    def __init__(self, mov):
        self.mov = mov  


class Aprendizagem:
    @abstractmethod
    def selecionar_acao(self, s: Estado) -> Acao:
        pass

    @abstractmethod
    def aprender(self, s: Estado, a: Acao, r: float, sn: Estado, an: Acao):
        pass


class MemoriaAprend:
    def __init__(self, valor_omissao=0.0):
        self.valor_omissao = valor_omissao
        self.memoria = {}

    @abstractmethod
    def atualizar(self, s: Estado, a: Acao, q: float):
        pass

    @abstractmethod
    def Q(self, s: Estado, a: Acao) -> float:
        pass


class MemoriaEsparsa(MemoriaAprend):
    def atualizar(self, s: Estado, a: Acao, q: float):
        self.memoria[(s.estado, a.mov)] = q

    def Q(self, s: Estado, a: Acao) -> float:
        return self.memoria.get((s.estado, a.mov), self.valor_omissao)


class SelAcao:
    def __init__(self, mem_aprend: MemoriaAprend, acoes: list[Acao], epsilon: float):
        self.mem_aprend = mem_aprend
        self.acoes = acoes
        self.epsilon = epsilon

    def seleccionar_acao(self, s: Estado) -> Acao:
        pass

    def max_acao(self, s: Estado) -> Acao:
        pass

class Egreedy(SelAcao):
    def max_acao(self, s: Estado) -> Acao:
        random.shuffle(self.acoes)
        return max(self.acoes, key=lambda a: self.mem_aprend.Q(s, a))

    def explorar(self) -> Acao:
        return random.choice(self.acoes)

    def selecionar_acao(self, s: Estado) -> Acao:
        if random.random() > self.epsilon:
            return self.max_acao(s)
        else:
            return self.explorar()


class AprendRef:
    def __init__(self, mem_aprend: MemoriaAprend, sel_acao: SelAcao, alfa: float, gama: float):
        self.mem_aprend = mem_aprend
        self.sel_acao = sel_acao
        self.alfa = alfa
        self.gama = gama

    @abstractmethod
    def aprender(self, s: Estado, a: Acao, r: float, sn: Estado, an: Acao = None):
        pass

class MecAprendRef(Aprendizagem):
    def __init__(self, acoes: list[Acao], mem_aprend: MemoriaAprend, sel_acao: SelAcao, aprend_ref: AprendRef):
        self.acoes = acoes
        self.mem_aprend = mem_aprend
        self.sel_acao = sel_acao
        self.aprend_ref = aprend_ref

    def aprender(self, s: Estado, a: Acao, r: float, sn: Estado, an: Acao = None):
        if (an is None):
            self.aprend_ref.aprender(s, a, r, sn) #an removido 
        else:
            self.aprend_ref.aprender(s, a, r, sn, an)    


    def selecionar_acao(self, s: Estado) -> Acao:
        return self.sel_acao.selecionar_acao(s)


class SARSA(AprendRef):

    def aprender(self, s: Estado, a: Acao, r: float, sn: Estado, an: Acao):
        qsa = self.mem_aprend.Q(s,a)
        qsnan = self.mem_aprend.Q(sn, an)
        q = qsa + self.alfa * (r + self.gama * qsnan - qsa )
        self.mem_aprend.atualizar(s, a, q)


class QLearning(AprendRef):
    def aprender(self, s, a, r, sn):
        an = self.sel_acao.max_acao(sn)
        qsa = self.mem_aprend.Q(s, a)
        qsnan = self.mem_aprend.Q(sn, an)
        q = qsa + self.alfa * (r + self.gama * qsnan - qsa)
        self.mem_aprend.atualizar(s, a, q)


class ModeloTR:

    def __init__(self):
        self.T = {}
        self.T = {}

    def atualizar(self, s, a, r, sn):
        self.T[(s,a)] = sn 
        self.R[(s,a)] = r

    def amostrar(self):
        s, a = random.choice(self.T.keys())
        sn = self.T[(s, a)]
        r = self.R[(s, a)]
        return s, a, r, sn

class DynaQ(QLearning):
    def __init__(self,mem_aprend, sel_acao, alfa,gama, num_sim):
        super().__init__(mem_aprend, sel_acao, alfa, gama)
        self.num_sim = num_sim
        self.modelo = ModeloTR() 

    def aprender(self, s, a, r, sn):
        super().aprender(s, a, r, sn)
        self.modelo.atualizar(s, a, r, sn)
        self.simular()

    def simular(self):
        for _ in range(self.num_sim):
            s, a, r, sn = self.modelo.amostar()
            super().aprender(s, a, r, sn)


#---------------------------Jogo do Galo------------------------------------

#Inspirado em https://medium.com/@ardra4/tic-tac-toe-using-q-learning-a-reinforcement-learning-approach-d606cfdd64a3


grelha = np.array([['-', '-', '-'],
                  ['-', '-', '-'],
                  ['-', '-', '-']])

jogadores = ['X', 'O']
numJogadores = len(jogadores)


#print da grelha
def printGrelha(grelha):
    for row in grelha:
        print(' | '.join(row))
        print('---------')


#converte a grelha para string
def convString(grelha):
    return ''.join(grelha.flatten())



#deteta se o jogo acabou ou não 
def fimJogo(grelha):

    for row in grelha:
        if len(set(row)) == 1 and row[0] != '-':
            return True, row[0]
        
    for col in grelha.T:                            #iteracao sobre a matriz tarnsposta desta forma é possivel iterar sobre as colunas da matriz como fossem linhas 
        if len(set(col)) == 1 and col[0] != '-':
            return True, col[0]   
    
    #Diagonais
    if len(set(grelha.diagonal())) == 1 and grelha[0, 0] != '-':
        return True, grelha[0, 0]
    if len(set(np.fliplr(grelha).diagonal())) == 1 and grelha[0, 2] != '-':
        return True, grelha[0, 2]
    
    # Empate
    if '-' not in grelha:
        return True, 'empate'
    
    return False, None    


#--------------------Teste-------------------------

memoria = MemoriaEsparsa()
acoes = [Acao((i, j)) for i in range(3) for j in range(3)]
sel_acao = Egreedy(memoria, acoes, epsilon=0.3)
aprend_ref = QLearning(memoria, sel_acao, alfa=0.5, gama=0.9)
#aprend_ref = SARSA(memoria, sel_acao, alfa=0.5, gama=0.9)
mec_aprend = MecAprendRef(acoes, memoria, sel_acao, aprend_ref)


def treino(episodios): #Qlearning

    for _ in range(episodios):
        grelha = np.array([['-', '-', '-'],
                          ['-', '-', '-'],
                          ['-', '-', '-']])
        estado_atual = Estado(convString(grelha))
        fim = False
        jogador = 0

        while not fim:
            acao = mec_aprend.selecionar_acao(estado_atual)
            mov = acao.mov
            if grelha[mov] == '-':
                grelha[mov] = jogadores[jogador]
                fim, vencedor = fimJogo(grelha)

                if fim:
                    if vencedor == jogadores[jogador]:
                         recompensa = 1 
                    elif vencedor == 'empate':
                        recompensa = 0.5
                    else:
                        recompensa = 0 

                    mec_aprend.aprender(estado_atual, acao, recompensa, Estado(convString(grelha)))
                
                if not fim:
                    prox_estado = Estado(convString(grelha))
                    mec_aprend.aprender(estado_atual, acao, 0, prox_estado)
                    estado_atual = prox_estado

                    jogador = 1 - jogador #troca de jogador

                    
'''
def treino(episodios):  # SARSA
    for _ in range(episodios):
        grelha = np.array([['-', '-', '-'],
                          ['-', '-', '-'],
                          ['-', '-', '-']])
        estado_atual = Estado(convString(grelha))
        acao_atual = mec_aprend.selecionar_acao(estado_atual)  
        fim = False
        jogador = 0

        while not fim:
            mov = acao_atual.mov
            if grelha[mov] == '-':
                grelha[mov] = jogadores[jogador]
                fim, vencedor = fimJogo(grelha)

                if fim:  
                    if vencedor == jogadores[jogador]:
                        recompensa = 1
                    elif vencedor == 'empate':
                        recompensa = 0.5
                    else:
                        recompensa = 0
                    mec_aprend.aprender(estado_atual, acao_atual, recompensa, Estado(convString(grelha)), None)
                else:  
                    prox_estado = Estado(convString(grelha))
                    prox_acao = mec_aprend.selecionar_acao(prox_estado)  
                    mec_aprend.aprender(estado_atual, acao_atual, 0, prox_estado, prox_acao)
                    estado_atual, acao_atual = prox_estado, prox_acao  
                    jogador = 1 - jogador
'''

#o algoritmo joga primeiro
def jogar(): #Qlearning

    grelha = np.array([['-', '-', '-'],
                      ['-', '-', '-'],
                      ['-', '-', '-']])
    jogador = 'O'  
    fim = False

    while not fim:
        printGrelha(grelha)
        print('          ')
        if jogador == 'X':
            i, j = map(int, input("Insiar a linha e coluna: ").split())
            acao = Acao((i, j))
        else:
            estado = Estado(convString(grelha))
            acao = mec_aprend.selecionar_acao(estado)

        mov = acao.mov
        if grelha[mov] == '-':
            grelha[mov] = jogador  
            fim, vencedor = fimJogo(grelha)  
            if fim:
                printGrelha(grelha)
                if vencedor == 'empate':
                    print("Empate")
                else:
                    if vencedor == 'O':
                        print('IA venceu')
                    else:
                        print('O jogador venceu')
                            

                    
            jogador = 'O' if jogador == 'X' else 'X'  #Alterar jogador


treino(10000)
print('Para jogar insira a linha e depois a coluna que pretenda separadas por um espaço exemplo (0 2)')
jogar()
    
   
       
    



