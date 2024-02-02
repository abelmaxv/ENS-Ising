import numpy as np
import matplotlib.pyplot as plt

class Ising:
    
    def __init__(self, n, beta):
        self.grid = 2*np.round(np.random.rand(n, n)) - 1
        self.beta = beta
        self.n = n
    def energy(self):
        n = self.n
        sum = 0
        for i in range(n):
            for j in range(n):
                sum += self.grid[i, j]*self.grid[(i+1)%n, j]
                sum += self.grid[i, j]*self.grid[i, (j+1)%n]
                sum += self.grid[i, j]*self.grid[(i-1)%n, j]
                sum += self.grid[i, j]*self.grid[i, (j-1)%n]
        return sum*(-self.beta)/2
    def mean(self):
        n = self.n
        sum = 0
        for i in range(n):
            for j in range(n):
                sum += self.grid[i, j]
        return sum/n**2
    def metropolis(self, N):
        n = self.n
        list = []
        for k in range(N):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            delta = -self.beta*2*(self.grid[i, j]*self.grid[(i+1)%n, j]+self.grid[i, j]*self.grid[i, (j+1)%n]+self.grid[i, j]*self.grid[(i-1)%n, j]+self.grid[i, j]*self.grid[i, (j-1)%n])
            if delta <= 0:
                self.grid[i, j] = -self.grid[i, j]
            else:
                x = np.random.random()
                if x < np.exp(-self.beta*delta):
                    self.grid[i, j] = -self.grid[i, j]
            list.append(self.mean())
        return list

if __name__ == "__main__" :
    tab = Ising(100, 0.6)
    print(tab.grid)
    y = tab.metropolis(100000)
    x = [i for i in range(100000)]
    plt.plot(x, y)
    plt.show()