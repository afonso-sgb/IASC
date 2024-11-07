import random
import numpy as np
from abc import abstractmethod


class Estado: 
    pass

class Acao:
    pass

class Aprendizagem:

    @abstractmethod
    def selecionar_acao(self, s: Estado) -> Acao:
        pass

    @abstractmethod
    def aprender(self, s: Estado, a: Acao, r:float, sn:Estado,an:Acao):
        pass


class MemoriaAprend:

    @abstractmethod
    def actualizar(s: Estado, a: Acao, q:float):
        pass

    @abstractmethod
    def Q(self, s: Estado, a: Acao) -> float:
        pass    

class MemoriaEsparsa(MemoriaAprend):

    def __init__(self, valor_omissao: float = 0.0):

        self.valor_omissao = valor_omissao
        self.memoria = {}

    def Q(self, s: Estado, a: Acao):

        return self.memoria.get((s, a), self.valor_omissao)    

    def atualizar(self, s: Estado, a: Acao, q: float):
        self.memoria[(s, a)] = q


class SelAcao:

    def __init__(self, mem_aprend: MemoriaAprend):
        self.mem_aprend = mem_aprend

    def seleccionar_acao(self, s: Estado) -> Acao: 
        pass

    def max_acao(self, s: Estado) -> Acao:
        pass


class Egreedy (SelAcao):

    def __init__(self, mem_aprend, acoes, epsilon):

        self.mem_aprend = mem_aprend
        self.acoes = acoes
        self.epsilion = epsilon

    def max_acao(self, s):
        random.shuffle(self.acoes)
        return np.argmax(self.acoes, lambda a : self.mem_aprend.Q(s, a))
        
    def aproveitar(self, s):
        return self.max_acao(s)
    
    def explorar(self):
        return random.choice(self.acoes)
    
    def selecionar_acao(self, s):
        if random.random() > self.epsilon:
            acao = self.aproveitar(s)
        else:
            acao = self.explorar()

        return acao       



class AprendRef:

    def __init__(self, mem_aprend: MemoriaAprend, sel_acao: SelAcao, alfa: float, gama: float):
    
        self.mem_aprend = mem_aprend
        self.sel_acao = sel_acao
        self.alfa = alfa
        self.gama = gama

    @abstractmethod
    def aprender(s: Estado, a: Acao, r:float, sn: Estado, an = None):
        pass


class SARSA(AprendRef):

    def aprender(self, s, a, r, sn, an):
        qsa = self.mem_aprend.Q(s,a)
        qsnan = self.mem_aprend.Q(sn, an)
        q = qsa + self.alfa * (r + self.gama * qsnan - qsa )
        self.mem_aprend.actualizar(s, a, q)


class QLearning(AprendRef):
    def aprender(self, s, a, r, sn):
        an = self.sel_acao.max_acao(sn)
        qsa = self.mem_aprend.Q(s,a)
        qsnan = self.mem_aprend.Q(sn, an)
        q = qsa + self.alfa * (r + self.gama * qsnan - qsa )
        self.mem_aprend.actualizar(s, a, q)   




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

'''
class QME(QLearning):

    def __init__(self, mem_aprend: MemoriaAprend, sel_acao: Selacao, alfa: float, gama: float, num_sim, dim_max):
        super().__init__(mem_aprend, sel_acao, alfa, gama)
        self.num_sim = num_sim
        self.memoria_experiencia = MemoriaExperiencia(dim_max)

    def aprender(self, s, a, r, sn):
        super().aprender(s, a, r, sn)
        e = (s, a, r, sn)

        self.memoria_experiencia.atualizar(e)
        self.simular()

    def simular(self):
        amostras = self.memoria_experiencia.amostrar(self.num_sim)
        for (s, a, r, sn) in range(amostras):
            super().aprender(s, a, r, sn)

class MemoriaExperiencia:

    def __init__(self, dim_max):
        self.dim_max = dim_max
        self.memoria = []

    def atualizar(self, e):
        if (self.memoria.dim() == self.dim_max):
            self.memoria.remove(0)
        self.memoria.append(e)

    def amostrar(self, n):
        n_amostras = min(n, self.memoria.dim())
        return random.sample(self.memoria, n_amostras)        

'''


class MecAprendRef:

    def __init__(self, acoes : Acao): #list[acao]?

        self.acoes = acoes
        self.memoria_aprend = MemoriaAprend()
        self.sel_acao = SelAcao(self.memoria_aprend)
        self.apren_ref = AprendRef(self.memoria_aprend, self.sel_acao, alfa, gama)

    def aprender(self, s: Estado, a: Acao, r: float, sn: Estado, an: Acao = None):

        self.apren_ref.aprender(s, a, r, sn, an)

       
    def selecionar_acao(self, s: Estado ) -> Acao:

        self.sel_acao.seleccionar_acao(s)
        
