import math
import numpy as np

class Quadratic_func:

    def __init__(self, q=None, max_iter=100, max_iter_hessian=100, hessian=False):
        self.q = q
        self.hessian = hessian
        self.max_iter = max_iter
        self.max_iter_hessian = max_iter_hessian

    def func(self, x):
        if x.size == 2:
            f = x.T.dot(self.q).dot(x)
            g = np.add(self.q, self.q.T).dot(x)
            if self.hessian:
                h = np.add(self.q, self.q.T)
                return f.item(), g, h
            else:
                return f.item(), g
        else:
            return np.einsum('ij,ji->i', x.T.dot(self.q), x)

class Rosenbrock_func:

    def __init__(self, max_iter=10000, max_iter_hessian=100, hessian=False):
        self.hessian = hessian
        self.max_iter = max_iter
        self.max_iter_hessian = max_iter_hessian

    def func(self, x):
        x1, x2 = x[0], x[1]
        if x.size == 2:
            f = 100*math.pow((x2-math.pow(x1, 2)), 2)+math.pow((1-x1), 2)
            df_x1 = -400*x1*(x2-math.pow(x1, 2))-2*(1-x1)
            df_x2 = 200*(x2-math.pow(x1, 2))
            df_x1_x1 = -400*x2+1200*math.pow(x1, 2)+2
            df_x2_x2 = 200
            df_x1_x2 = -400*x1
            df_x2_x1 = df_x1_x2
            g = np.array([df_x1, df_x2]).T
            if self.hessian:
                h11 = df_x1_x1
                h12 = df_x1_x2
                h21 = df_x2_x1
                h22 = df_x2_x2
                h = np.array([[h11, h12], [h21, h22]])
                return f, g, h
            else:
                return f, g
        else:
            return 100*(x2-x1**2)**2+(1-x1)**2

class Linear_func:

    def __init__(self, a=None, max_iter=100, max_iter_hessian=100, hessian=False):
        self.a = a
        self.hessian = hessian
        self.max_iter = max_iter
        self.max_iter_hessian = max_iter_hessian

    def func(self, x):
        if x.size == 2:
            f = self.a.T.dot(x)
            g = self.a.T
            if self.hessian:
                h = np.zeros((2, 2))
                return f, g, h
            else:
                return f, g
        else:
            return self.a.T.dot(x)

class Exponential_func:

    def __init__(self, max_iter=100, max_iter_hessian=100, hessian=False):
        self.hessian = hessian
        self.max_iter = max_iter
        self.max_iter_hessian = max_iter_hessian

    def func(self, x=None):
        x1, x2 = x[0], x[1]
        if x.size == 2:
            f = math.exp(x1+3*x2-0.1)+math.exp(x1-3*x2-0.1)+math.exp(-x1-0.1)
            df_x1 = math.exp(x1+3*x2-0.1)+math.exp(x1-3*x2-0.1)-math.exp(-x1-0.1)
            df_x2 = 3*math.exp(x1+3*x2-0.1)-3*math.exp(x1-3*x2-0.1)
            df_x1_x1 = math.exp(x1+3*x2-0.1)+math.exp(x1-3*x2-0.1)+math.exp(-x1-0.1)
            df_x2_x2 = 9*math.exp(x1+3*x2-0.1)+9*math.exp(x1-3*x2-0.1)
            df_x1_x2 = 3*math.exp(x1+3*x2-0.1)-3*math.exp(x1-3*x2-0.1)
            df_x2_x1 = df_x1_x2
            g = np.array([df_x1, df_x2]).T
            if self.hessian:
                h11 = df_x1_x1
                h12 = df_x1_x2
                h21 = df_x2_x1
                h22 = df_x2_x2
                h = np.array([[h11, h12], [h21, h22]])
                return f, g, h
            else:
                return f, g
        else:
            return np.exp(x1+3*x2-0.1)+np.exp(x1-3*x2-0.1)+np.exp(-x1-0.1)

# Functions
q1 = Quadratic_func(q=np.array([[1, 0], [0, 1]]))
q2 = Quadratic_func(q=np.array([[1, 0], [0, 100]]))
q3 = Quadratic_func(q=np.array([[(3**0.5)/2, -0.5], [0.5, (3**0.5)/2]]).T.dot(np.array([[100, 0], [0, 1]])).dot(np.array([[(3**0.5)/2, -0.5], [0.5, (3**0.5)/2]])))
f_rosenbrock = Rosenbrock_func()
f_linear = Linear_func(a=np.array([1, 1]))
f_exponential = Exponential_func()
