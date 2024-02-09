import numpy as np
import matplotlib.pyplot as plt

class Ising:
    """ Simulation of the Ising model """
    
    def __init__(self, n, beta):
        self.config = 2*np.round(np.random.rand(n, n)) - 1
        self.beta = beta
        self.size = n
        self.energy = self.hamiltonian()
    
    def randomize(self): 
         self.config = 2*np.round(np.random.rand(self.size, self.size)) - 1

    def hamiltonian(self): #TO IMPROVE (with convolution)  ! 
        n = self.size
        sum = 0
        for i in range(n):
            for j in range(n):
                sum += self.config[i, j]*self.config[(i+1)%n, j]
                sum += self.config[i, j]*self.config[i, (j+1)%n]
                sum += self.config[i, j]*self.config[(i-1)%n, j]
                sum += self.config[i, j]*self.config[i, (j-1)%n]
        return sum*(-self.beta)/2
        
    def mean_spin(self):
        n = self.size
        sum = 0
        for i in range(n):
            for j in range(n):
                sum += self.grid[i, j]
        return sum/n**2


    def metropolis(self, N):
        """
        Generates a configuration of the Ising model following the distribution \mu_{\beta} with the metropolis algorithm
        """
        n = self.size
        for k in range(N):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            spin_i = self.config[i,j]

            delta_E = self.beta*2*spin_i*(self.config[(i+1)%n,j] + self.config[(i-1)%n,j] + self.config[i,(j+1)%n] + self.config[i, (j-1)%n])
            if delta_E < 0 :
                self.config[i, j] = (-1)*spin_i
                self.energy += delta_E
            else:
                x = np.random.random()
                if x < np.exp(-self.beta*delta_E):
                    self.config[i, j] = (-1)*spin_i
                    self.energy += delta_E
    
    def display(self):
        plt.imshow(self.config)
        plt.show()

if __name__ == "__main__" :
    iter = 10000000
    size = 1000
    beta = 2000

    model = Ising(size, beta)
    model.metropolis(iter)
    model.display()