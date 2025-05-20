import scipy.integrate as integrate
import dionysus as dio
import numpy as np
import matplotlib.pyplot as plt
import collections.abc

class Landscape(object):
    def __init__(self, f = None, prime = 47, max_dim = 2, max_t = None, n_evaluations = 1000):
        self.max_dim = max_dim
        self.computed_norm = None
        self.computed_norm_limit = 0
        self.prime = prime
        self.f = f

        if max_t is None:
            self.max_t = 0
            for p in f:
                self.max_t = max(self.max_t, p.data)
        else:
            self.max_t = max_t

        if f is not None:
            p = dio.cohomology_persistence(f, prime, True)
            self.dgms = dio.init_diagrams(p, f)
            self.points = [[(p.birth, p.death) for p in self.dgms[p]] for p in range(self.max_dim+1)]

            self.evaluate_on_interval(n_evaluations)

    def evaluate_on_interval(self, n):
        x_axis = np.linspace(0, self.max_t, n)
        self.evaluations = [np.array([self(t, p=dim) for t in x_axis]) for dim in range(self.max_dim+1)]
        return self.evaluations

    def __call__(self, t, p=0):
        return np.sum([max(t-b, 0) if t<(b+d)/2 else max(d-t, 0) for b, d in self.points[p] if d != np.inf])
    
    def norm(self, return_error=False, limit=1000, use_evaluations=True):
        if use_evaluations:
            integral = 0
            for p in range(self.max_dim+1):
                integral += integrate.trapezoid(self.evaluations[p]**2, dx=self.max_t/len(self.evaluations[p]))
            return integral
        if self.computed_norm_limit < limit:
            self.computed_norm = 0
            self.norm_error = 0
            for p in range(self.max_dim+1):
                integrate_result = integrate.quad(lambda t: self(t, p=p)**2, 0, self.max_t, limit=limit)
                self.computed_norm += integrate_result[0]
                self.norm_error += integrate_result[1]
        if return_error:
            return self.computed_norm, self.norm_error
        return self.computed_norm
        
    def compute_statistic(self, land_list, limit=1000):
        if isinstance(land_list, list):
            difference = 0
            error = 0
            sum_landscape = np.sum([land for land in land_list])
            n_others = len(land_list)

            for p in range(self.max_dim+1):
                integrate_result = integrate.quad(lambda t: (self(t, p=p) - 1/n_others*sum_landscape(t, p=p))**2, 0, max(self.max_t, *[land.max_t for land in land_list]), limit=limit)
                difference += integrate_result[0]
                error += integrate_result[1]
            print("Error: ", error)
            return difference
    
    def __add__(self, other):
        if self.max_t != other.max_t:
            raise ValueError("Landscapes must have the same max_t")
        for dim in range(min(self.max_dim, other.max_dim) + 1):
            if len(self.evaluations[dim]) != len(other.evaluations[dim]):
                raise ValueError("Landscapes must have the same number of evaluations")
        
        new_land = Landscape(max_t = self.max_t, max_dim=min(self.max_dim, other.max_dim))
        new_land.evaluations = [self.evaluations[p] + other.evaluations[p] for p in range(new_land.max_dim+1)]
        return new_land
    
    def __sub__(self, other):
        if self.max_t != other.max_t:
            raise ValueError("Landscapes must have the same max_t")
        for dim in range(min(self.max_dim, other.max_dim) + 1):
            if len(self.evaluations[dim]) != len(other.evaluations[dim]):
                raise ValueError("Landscapes must have the same number of evaluations")
        
        new_land = Landscape(max_t = self.max_t, max_dim=min(self.max_dim, other.max_dim))
        new_land.evaluations = [self.evaluations[p] - other.evaluations[p] for p in range(new_land.max_dim+1)]
        return new_land
    
    def __mult__(self, other):
        if np.isscalar(other):
            new_land = Landscape(max_t = self.max_t, max_dim=self.max_dim)
            new_land.evaluations = [self.evaluations[p] * other for p in range(self.max_dim+1)]
            return new_land
        else:
            raise ValueError("Multiplication is only supported with scalars")
    
    def __rmult__(self, other):
        if np.isscalar(other):
            new_land = Landscape(max_t = self.max_t, max_dim=self.max_dim)
            new_land.evaluations = [self.evaluations[p] * other for p in range(self.max_dim+1)]
            return new_land
        else:
            raise ValueError("Multiplication is only supported with scalars")
        
    def __truediv__(self, other):
        if isinstance(self, Landscape):
            if np.isscalar(other):
                new_land = Landscape(max_t = self.max_t, max_dim=self.max_dim)
                new_land.evaluations = [self.evaluations[p] / other for p in range(self.max_dim+1)]
                return new_land
            else:
                print("other type:", type(other))
                raise ValueError("Division is only supported with scalars")
        elif isinstance(other, Landscape):
            if np.isscalar(self):
                new_land = Landscape(max_t = other.max_t, max_dim=other.max_dim)
                new_land.evaluations = [other.evaluations[p] / self for p in range(other.max_dim+1)]
                return new_land
            else:
                raise ValueError("Division is only supported with scalars")
        
    def __rtruediv__(self, other):
        if np.isscalar(other):
            new_land = Landscape(max_t = self.max_t, max_dim=self.max_dim)
            new_land.evaluations = [other / self.evaluations[p] for p in range(self.max_dim+1)]
            return new_land
        else:
            raise ValueError("Division is only supported with scalars")

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        for dim in range(self.max_dim+1):
            ax.plot(np.linspace(0, self.max_t, len(self.evaluations[dim])), self.evaluations[dim], c={0: "b", 1: "r", 2: "g", 3: "y"}[dim])
        
        ax.legend([f"Dim {i}" for i in range(self.max_dim+1)])
        ax.set_xlabel("t")
        ax.set_ylabel("Landscape")
        ax.set_title("Landscape")