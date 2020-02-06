import numpy as np
import matplotlib.pyplot as plt

class ProbabilityModel:

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self, size=[]):
        return np.random.uniform(size=size)


# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
    
    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma

    # Box-Muller Transform
    def sample(self):
        pm = ProbabilityModel()
        u1 = pm.sample()
        u2 = pm.sample()
        y = self.mu + np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2) * self.sigma
        return y

    def draw(self):
        res = []
        for i in range (-10000,10000):
            res.append(self.sample())
        plt.hist(res)
        plt.show()
    
# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self,Mu,Sigma):
        self.mu = Mu
        self.sigma = Sigma
        self.K = np.linalg.cholesky(Sigma)
        self.shape = np.shape(Mu)

    def sample(self):
        pm = ProbabilityModel()
        u1 = pm.sample(size=self.shape)
        u2 = pm.sample(size=self.shape)
        y = self.mu + np.matmul(self.K, np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2))
        return y

    def draw(self):
        X = []
        Y = []
        for i in range (0,10000):
            res = self.sample()
            X.append(res[0])
            Y.append(res[1])
        plt.scatter(X, Y)
        plt.show()

# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self,ap):
        self.ap = ap

    def sample(self):
        x = ProbabilityModel().sample()
        cumsum = np.cumsum(self.ap)
        y = [i for i, num in enumerate(cumsum) if num > x][0]
        return y

    def draw(self):
        res = []
        for i in range(10000):
            res.append(self.sample())
        plt.hist(res)
        plt.show()

# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
    
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self,ap,pm):
        self.ap = ap
        self.pm = pm
        self.n = len(ap)

    def sample(self):
        # return sum([self.ap[i]*self.pm[i].sample() for i in range(self.n)])
        i = Categorical(self.ap).sample()
        return self.pm[i].sample()
        
    def prob(self):
        cnt = 0
        for i in range(10000):
            s = self.sample()
            if (s[0]-0.1)*(s[0]-0.1) + (s[1]-0.2)*(s[1]-0.2) < 1.0:
                cnt += 1
        return cnt/10000

if __name__ == "__main__":
    # model = Categorical([0.1,0.1,0.3,0.3,0.2])
    # model.draw()
    # model = UnivariateNormal(0, 1)
    # model.draw()
    # model = MultiVariateNormal([1,1], [[1,0.5],[0.5,1]])
    # model.draw()
    pms = (MultiVariateNormal([-1,-1], [[1,0],[0,1]]), 
        MultiVariateNormal([-1,1], [[1,0],[0,1]]), 
        MultiVariateNormal([1,-1], [[1,0],[0,1]]), 
        MultiVariateNormal([1,1], [[1,0],[0,1]]))
    model = MixtureModel([0.25,0.25,0.25,0.25], pms)
    print(model.prob())
