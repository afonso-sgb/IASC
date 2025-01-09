import numpy as np


ERROS = []

def sig(x):
    return 1 / (1 + np.exp(-x))


def deriv(f, x):
    h = 1e-6
    return (f(x + h) - f(x)) / h


class CamadaEntrada:
    def __init__(self, ds):
        self.ds = ds
        self.y = [0 for _ in range(ds)]

    def propagar(self, x):
        self.y = x  
        return self.y


class Neuronio:
#OK
    def __init__(self, d, phi):

        self.phi = phi
        self.w = np.random.uniform(-1, 1, d)  
        self.b = np.random.uniform(-1, 1)  

        self.delta_w = np.zeros(d) 
        self.delta_b = 0
        self.h = 0
        self.y = 0
        self.y_linha = 0


    def propagar(self, x):

        #x = np.array(x)  

        self.h = np.dot(self.w, x) + self.b 
        self.y = self.phi(self.h)

        self.y_linha = deriv(self.phi, self.h)
        return self.y

    def adaptar(self, delta, y_menos_1, alpha, beta): #alpha taxa de aprendizagem

        M_w = beta * self.delta_w  
        self.delta_w = -alpha * self.y_linha * np.array(y_menos_1) * delta + M_w

        self.w += self.delta_w
        M_b = beta * self.delta_b
        self.delta_b = -alpha * self.y_linha * delta + M_b
        self.b += self.delta_b

#ok

class CamadaDensa:

    def __init__(self, de, ds, phi):

        self.de = de
        self.ds = ds
        self.phi = phi
        self.y = []
        self.neuronios = [Neuronio(self.de, self.phi) for _ in range(self.ds)]

   # @property
    #def y(self):
     #   return [neuronio.y for neuronio in self.neuronios]

    def propagar(self, x):
        
        self.y = [neuronio.propagar(x) for neuronio in self.neuronios]
        return self.y

    def adaptar(self, delta_n, y_n_menos_1, alpha, beta):
        for j in range(self.ds):
            self.neuronios[j].adaptar(delta_n[j], y_n_menos_1, alpha, beta)


class RedeNeuronal:

    def __init__(self, forma, phi):

        self.forma = forma
        self.phi = phi
        self.camadas = []
        self.N = len(forma)

        
        ds_1 = forma[0]  
        camada_1 = CamadaEntrada(ds_1)
        self.camadas.append(camada_1)

        
        for n in range(1, self.N):
            de_n = forma[n - 1]
            ds_n = forma[n]
            camada_n = CamadaDensa(de_n, ds_n, self.phi)
            self.camadas.append(camada_n)

    def delta_saida(self, y_n, y):
        return [y_n[k] - y[k] for k in range(len(y))]

    def retopropagar(self, delta_n, alpha, beta):
        delta = delta_n
        #dimensao = len(self.forma)

        for n in range(len(self.forma) - 1, 0, -1):
            y_n_menos_1 = self.camadas[n-1].y
            d_n_menos_1 = self.camadas[n-1].ds
            d_n = self.camadas[n].ds
            neur_n = self.camadas[n].neuronios
            delta_n_menos_1 = []

            for i in range(d_n_menos_1):
                somat = 0
                for j in range(d_n):
                    somat += neur_n[j].w[i] * delta[j] * neur_n[j].y_linha
                delta_n_menos_1.append(somat)
            self.camadas[n].adaptar(delta, y_n_menos_1, alpha, beta)
            delta = delta_n_menos_1        


    def adaptar(self, x, y, alpha, beta):
        y_n = self.propagar(x)
        delta_n = self.delta_saida(y_n, y)
        self.retopropagar(delta_n, alpha, beta)
        self.K = len(delta_n)

        soma = 0
        for k in range(self.K):
            soma += pow(delta_n[k], 2)
        erro = soma /self.K 
        return erro    


        #erro = (1 / K) * sum((delta_n[k]) ** 2 for k in range(K))
        #return erro

    def treinar(self, X, Y, n_epocas, erro_max, alpha, beta):
        for _ in range(n_epocas):
            erro = 0
            for x, y in zip(X, Y):
                erro_x = self.adaptar(x, y, alpha, beta)
                erro = max(erro, erro_x)
            ERROS.append(erro)    
            if (erro <= erro_max):
                break

    def prever(self, X):
        Y = [self.propagar(x) for x in X]
        return Y

    def propagar(self, x):
        y = x
        for camada in self.camadas:
            y = camada.propagar(y)
        return y



