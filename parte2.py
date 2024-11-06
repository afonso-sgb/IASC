import random
import numpy as np
from abc import abstractmethod


class Estado: 
    pass

class Accao:
    pass

class Aprendizagem:

    @abstractmethod
    def selecionar_accao(self, s: Estado) -> Accao:
        pass

    @abstractmethod
    def aprender(self, s: Estado, a: Accao, r:float, sn:Estado,an:Accao):
        pass


class MemoriaAprend:

    def actualizar(s: Estado, a: Accao, q:float):
        pass

    def Q(self, s: Estado, a: Accao) -> float:
        pass    

class MemoriaEsparsa(MemoriaAprend):

    def __init__(self, valor_omissao: float = 0.0):

        self.valor_omissao = valor_omissao
        self.memoria = {}

    def Q(self, s: Estado, a: Accao):

        return self.memoria.get((s, a), self.valor_omissao)    

    def atualizar(self, s: Estado, a: Accao, q: float):
        self.memoria[(s, a)] = q


class SelAccao:

    def __init__(self, mem_aprend: MemoriaAprend):
        self.mem_aprend = mem_aprend

    def seleccionar_accao(self, s: Estado) -> Accao: 
        pass

    def max_accao(self, s: Estado) -> Accao:
        pass


class Egreedy (SelAccao):

    def __init__(self, mem_aprend, accoes, epsilon):

        self.mem_aprend = mem_aprend
        self.accoes = accoes
        self.epsilion = epsilon

    def max_accao(self, s):
        random.shuffle(self.accoes)
        return np.argmax(self.accoes, lambda a : self.mem_aprend.Q(s, a))
        
    def aproveitar(self, s):
        return self.max_accao(s)
    
    def explorar(self):
        return random.choice(self.accoes)
    
    def selecionar_accao(self, s):
        if random.random() > self.epsilon:
            accao = self.aproveitar(s)
        else:
            accao = self.explorar()

        return accao       



class AprendRef:

    def __init__(self, mem_aprend: MemoriaAprend, sel_accao: SelAccao, alfa: float, gama: float):
    
        self.mem_aprend = mem_aprend
        self.sel_accao = sel_accao
        self.alfa = alfa
        self.gama = gama

    @abstractmethod
    def aprender(s: Estado, a: Accao, r:float, sn: Estado, an = None):
        pass


class SARSA(AprendRef):

    def aprender(self, s, a, r, sn, an):
        qsa = self.mem_aprend.Q(s,a)
        qsnan = self.mem_aprend.Q(sn, an)
        q = qsa + self.alfa * (r + self.gama * qsnan - qsa )
        self.mem_aprend.actualizar(s, a, q)


class QLearning(AprendRef):
    def aprender(self, s, a, r, sn):
        an = self.sel_accao.max_accao(sn)
        qsa = self.mem_aprend.Q(s,a)
        qsnan = self.mem_aprend.Q(sn, an)
        q = qsa + self.alfa * (r + self.gama * qsnan - qsa )
        self.mem_aprend.actualizar(s, a, q)   




class DynaQ(QLearning):
    def __init__(self,mem_aprend, sel_accao, alfa,gama, num_sim):
        super().__init__(mem_aprend, sel_accao, alfa, gama)
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


class QME(QLearning):

    def __init__(self, mem_aprend: MemoriaAprend, sel_accao: SelAccao, alfa: float, gama: float, num_sim, dim_max):
        super().__init__(mem_aprend, sel_accao, alfa, gama)
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




class MecAprendRef:

    def __init__(self, accoes : Accao): #list[Accao]?
        self.accoes = accoes

    def aprender(s: Estado, a: Accao, r: float, sn: Estado, an):
        pass

    def selecionar_accao(self, s: Estado ) -> Accao:
        pass



        
    

        



