import scipy.integrate as integrate
import dionysus as dio
import numpy as np
import matplotlib.pyplot as plt
import collections.abc
from Laplacian_Functions_torch import *

class Landscape(object):
    def __init__(self, f = None, prime = 47, max_dim = 1, max_t = None, n_evaluations = 1000, show_diagram=False, use_infinite = False):
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
            self.points = [[(p.birth, p.death) for p in self.dgms[q]] for q in range(self.max_dim+1)]
            if show_diagram:
                dio.plot.plot_all_diagrams(self.dgms, show=True, labels=True)

            self.evaluate_on_interval(n_evaluations, use_infinite=use_infinite)

    def show_diagram(self, show=True, limits=None):
        if self.f is not None and self.dgms is None:
            p = dio.cohomology_persistence(self.f, self.prime, True)
            self.dgms = dio.init_diagrams(p, self.f)
            # self.points = [[(p.birth, p.death) for p in self.dgms[q]] for q in range(self.max_dim+1)]
        if self.dgms is not None:
            dio.plot.plot_all_diagrams(self.dgms, show=show, labels=True, limits=limits)
        else:
            print("No diagram to show. Please add filtration first.")

    def evaluate_on_interval(self, n, use_infinite = False):
        x_axis = np.linspace(0, self.max_t, n)

        self.evaluations = {q: {k: [] for k in range(len(self.points[q]))} for q in range(self.max_dim+1)}
        for t in x_axis:
            for q in range(self.max_dim+1):
                if not use_infinite:
                    evaluation_for_every_k = np.array([max(t-b, 0) if t<(b+d)/2 else max(d-t, 0) for b, d in self.points[q] if d != np.inf])
                else:
                    evaluation_for_every_k = np.array([max(t-b, 0) if t<(b+d)/2 else max(d-t, 0) for b, d in self.points[q] if d != np.inf] + 
                                                      [max(t-b, 0) for b, d in self.points[q] if d == np.inf])
                evaluation_for_every_k[::-1].sort()
                for k in range(len(self.points[q])):
                    if k < len(evaluation_for_every_k):
                        self.evaluations[q][k].append(evaluation_for_every_k[k])
                    else:
                        self.evaluations[q][k].append(0)
        
        for q in range(self.max_dim+1):
            for k in range(len(self.points[q])):
                self.evaluations[q][k] = np.array(self.evaluations[q][k])
        # self.evaluations = [np.array([self(t, p=dim) for t in x_axis]) for dim in range(self.max_dim+1)]
        return self.evaluations

    def __call__(self, t, k, p=0):
        return np.partition([max(t-b, 0) if t<(b+d)/2 else max(d-t, 0) for b, d in self.points[p] if d != np.inf], -(k+1))[-(k+1)]
    
    def norm(self, return_error=False, limit=1000, use_evaluations=True, max_k=None, dim = None):
        if use_evaluations:
            integral = 0
            if dim is None:
                q_bar = range(self.max_dim+1)
            else:
                q_bar = [dim]
            for q in q_bar:
                if max_k is None:
                    for k in range(len(self.evaluations[q].keys())):
                        integral += integrate.trapezoid(self.evaluations[q][k]**2, dx=self.max_t/len(self.evaluations[q][k]))
                else:
                    for k in range(min(max_k, len(self.evaluations[q].keys()))):
                        integral += integrate.trapezoid(self.evaluations[q][k]**2, dx=self.max_t/len(self.evaluations[q][k]))
            return integral
        if self.computed_norm_limit < limit:
            self.computed_norm = 0
            self.norm_error = 0
            for p in range(self.max_dim+1):
                for k in range(len(self.points[p])):
                    integrate_result = integrate.quad(lambda t: self(t, k, p=p)**2, 0, self.max_t, limit=limit)
                    self.computed_norm += integrate_result[0]
                    self.norm_error += integrate_result[1]
        if return_error:
            return self.computed_norm, self.norm_error
        return self.computed_norm
        
    def sum_k(self):
        new_land = Landscape(max_t = self.max_t, max_dim=self.max_dim)
        new_land.evaluations = {q: {} for q in range(self.max_dim+1)}
        for q in range(self.max_dim+1):
            k_combined = [self.evaluations[q][k] for k in range(len(self.evaluations[q].keys()))]
            if len(k_combined) > 0:
                new_land.evaluations[q][0] = np.sum(k_combined, axis=0)
            else:
                new_land.evaluations[q][0] = np.zeros(len(new_land.evaluations[q-1][0]))
        return new_land
    # def compute_statistic(self, land_list, limit=1000):
    #     if isinstance(land_list, list):
    #         difference = 0
    #         error = 0
    #         sum_landscape = np.sum([land for land in land_list])
    #         n_others = len(land_list)

    #         for p in range(self.max_dim+1):
    #             integrate_result = integrate.quad(lambda t: (self(t, p=p) - 1/n_others*sum_landscape(t, p=p))**2, 0, max(self.max_t, *[land.max_t for land in land_list]), limit=limit)
    #             difference += integrate_result[0]
    #             error += integrate_result[1]
    #         print("Error: ", error)
    #         return difference
    
    def __add__(self, other):
        if self.max_t != other.max_t:
            raise ValueError("Landscapes must have the same max_t")

        new_land = Landscape(max_t = self.max_t, max_dim=min(self.max_dim, other.max_dim))

        new_land.evaluations = {q: {k: self.evaluations[q][k] + other.evaluations[q][k] for k in range(min(len(other.evaluations[q].keys()), len(self.evaluations[q].keys())))} for q in range(new_land.max_dim+1)}
        
        for q in range(new_land.max_dim+1):
            if len(self.evaluations[q]) == 0:
                new_land.evaluations[q] = other.evaluations[q]
            elif len(other.evaluations[q]) == 0:
                new_land.evaluations[q] = self.evaluations[q]
            elif len(self.evaluations[q][0]) != len(other.evaluations[q][0]):
                raise ValueError("Landscapes must have the same number of evaluations")
            
            if len(self.evaluations[q].keys()) < len(other.evaluations[q].keys()):
                for k in range(len(self.evaluations[q].keys()), len(other.evaluations[q].keys())):
                    new_land.evaluations[q][k] = other.evaluations[q][k]
            elif len(self.evaluations[q].keys()) > len(other.evaluations[q].keys()):
                for k in range(len(other.evaluations[q].keys()), len(self.evaluations[q].keys())):
                    new_land.evaluations[q][k] = self.evaluations[q][k]
        # new_land.evaluations = [self.evaluations[p] + other.evaluations[p] for p in range(new_land.max_dim+1)]
        return new_land
    
    def __sub__(self, other):
        if self.max_t != other.max_t:
            raise ValueError("Landscapes must have the same max_t")
        
        new_land = Landscape(max_t = self.max_t, max_dim=min(self.max_dim, other.max_dim))

        new_land.evaluations = {q: {k: self.evaluations[q][k] - other.evaluations[q][k] for k in range(min(len(other.evaluations[q].keys()), len(self.evaluations[q].keys())))} for q in range(new_land.max_dim+1)}
        
        for q in range(new_land.max_dim+1):
            if len(self.evaluations[q]) == 0:
                for k in range(len(other.evaluations[q].keys())):
                    new_land.evaluations[q][k] = -other.evaluations[q][k]
            elif len(other.evaluations[q]) == 0:
                for k in range(len(self.evaluations[q].keys())):
                    new_land.evaluations[q][k] = self.evaluations[q][k]
            elif len(self.evaluations[q][0]) != len(other.evaluations[q][0]):
                raise ValueError("Landscapes must have the same number of evaluations")
            
            if len(self.evaluations[q].keys()) < len(other.evaluations[q].keys()):
                for k in range(len(self.evaluations[q].keys()), len(other.evaluations[q].keys())):
                    new_land.evaluations[q][k] = -other.evaluations[q][k]
            elif len(self.evaluations[q].keys()) > len(other.evaluations[q].keys()):
                for k in range(len(other.evaluations[q].keys()), len(self.evaluations[q].keys())):
                    new_land.evaluations[q][k] = self.evaluations[q][k]

        # new_land.evaluations = [self.evaluations[p] - other.evaluations[p] for p in range(new_land.max_dim+1)]
        return new_land
    
    def __mult__(self, other):
        if np.isscalar(other):
            new_land = Landscape(max_t = self.max_t, max_dim=self.max_dim)
            new_land.evaluations = {q: {k: self.evaluations[q][k] * other for k in range(len(self.evaluations[q].keys()))} for q in range(self.max_dim+1)}
            # new_land.evaluations = [self.evaluations[p] * other for p in range(self.max_dim+1)]
            return new_land
        else:
            raise ValueError("Multiplication is only supported with scalars")
    
    def __rmult__(self, other):
        if np.isscalar(other):
            new_land = Landscape(max_t = self.max_t, max_dim=self.max_dim)
            new_land.evaluations = {q: {k: self.evaluations[q][k] * other for k in range(len(self.evaluations[q].keys()))} for q in range(self.max_dim+1)}
            # new_land.evaluations = [self.evaluations[p] * other for p in range(self.max_dim+1)]
            return new_land
        else:
            raise ValueError("Multiplication is only supported with scalars")
        
    def __truediv__(self, other):
        if isinstance(self, Landscape):
            if np.isscalar(other):
                new_land = Landscape(max_t = self.max_t, max_dim=self.max_dim)
                new_land.evaluations = {q: {k: self.evaluations[q][k] / other for k in range(len(self.evaluations[q].keys()))} for q in range(self.max_dim+1)}
                return new_land
            else:
                print("other type:", type(other))
                raise ValueError("Division is only supported with scalars")
        elif isinstance(other, Landscape):
            if np.isscalar(self):
                new_land = Landscape(max_t = other.max_t, max_dim=other.max_dim)
                new_land.evaluations = {q: {k: other.evaluations[q][k] / self for k in range(len(other.evaluations[q].keys()))} for q in range(other.max_dim+1)}
                return new_land
            else:
                raise ValueError("Division is only supported with scalars")
        
    def __rtruediv__(self, other):
        if np.isscalar(other):
            new_land = Landscape(max_t = other.max_t, max_dim=other.max_dim)
            new_land.evaluations = {q: {k: other.evaluations[q][k] / self for k in range(len(other.evaluations[q].keys()))} for q in range(other.max_dim+1)}
            # new_land.evaluations = [other / self.evaluations[p] for p in range(self.max_dim+1)]
            return new_land
        else:
            raise ValueError("Division is only supported with scalars")

    def plot(self, ax=None, max_k=3, limits=None):
        if ax is None:
            fig, ax = plt.subplots(1, self.max_dim+1, figsize=(5*(self.max_dim+1), 5))
            
        for dim in range(self.max_dim+1):
            for k in range(min(len(self.evaluations[dim].keys()), max_k)):
                # max_non_zero = len(self.evaluations[dim])
                # for i in range(len(self.evaluations[dim])-1, -1, -1):
                #     if self.evaluations[dim][k][i] != 0:
                #         max_non_zero = i+1
                #         break
                # ax[dim].plot(np.linspace(0, self.max_t, len(self.evaluations[dim][k]))[:max_non_zero], self.evaluations[dim][k][:max_non_zero], c={0: "b", 1: "r", 2: "g", 3: "y"}[dim])
                ax[dim].plot(np.linspace(0, self.max_t, len(self.evaluations[dim][k])), self.evaluations[dim][k])#, c={0: "b", 1: "r", 2: "g", 3: "y"}[dim])
            

            if limits is not None:
                if isinstance(limits[0], list):
                    if len(limits) > dim-self.min_dim:
                        ax[dim].set_xlim(limits[dim-self.min_dim][0], limits[dim-self.min_dim][1])
                        ax[dim].set_ylim(limits[dim-self.min_dim][2], limits[dim-self.min_dim][3])
                else:
                    ax[dim].set_xlim(limits[0], limits[1])
                    ax[dim].set_ylim(limits[2], limits[3])
            # if limits is not None:
            #     ax[dim].set_xlim(limits[0], limits[1])
            #     ax[dim].set_ylim(limits[2], limits[3])

            ax[dim].legend([f"k={k}" for k in range(min(len(self.evaluations[dim].keys()), max_k))])
            ax[dim].set_xlabel("t")
            ax[dim].set_ylabel("Landscape")
            ax[dim].set_title(f"Landscape dim {dim}")
        return ax

class Lap_Landscape(object):
    def __init__(self, f = None, min_dim = 0, max_dim = 1, max_t = None, n_evaluations = 1000, show_trace_diagram=False, Laplacian_fun = None,
                 compute_only_trace=False):
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.computed_norm = None
        self.computed_norm_limit = 0
        self.f = f
        self.Laplacian_fun = Laplacian_fun
        self.compute_only_trace = compute_only_trace

        if max_t is None:
            self.max_t = 0
            for p in f:
                self.max_t = max(self.max_t, p.data)
        else:
            self.max_t = max_t

        if self.f is not None:
            p = dio.cohomology_persistence(self.f, 47, True)
            self.dgms = dio.init_diagrams(p, self.f)
            self.eigenvalues, self.relevant_times = cross_Laplaican_eigenvalues_less_memory(f, device="cpu", Laplacian_fun=Laplacian_fun, min_dim=min_dim, max_dim=max_dim, compute_only_trace=compute_only_trace)
            s_lists, t_lists, cmaps = plot_trace_diagram(self.f, self.eigenvalues, show=show_trace_diagram, max_dim=max_dim, min_dim=min_dim, title="Laplacian diagram")
            self.points = [[(s_lists[q][point_i], t_lists[q][point_i], cmaps[q][point_i]) for point_i in range(len(s_lists[q]))] for q in range(self.max_dim+1)]

            
            self.evaluate_on_interval(n_evaluations)
        else:
            self.eigenvalues = None
            self.points = None

    def show_trace_diagram(self, show=True, lap_pt_style=None):
        if self.dgms is not None and self.eigenvalues is not None:
            plot_trace_diagram_no_f(self.dgms, self.eigenvalues, show=show, min_dim=self.min_dim, max_dim=self.max_dim, lap_pt_style=lap_pt_style, title="Laplacian diagram")
        elif self.f is not None and self.Laplacian_fun is not None:
                if self.eigenvalues is None:
                    self.eigenvalues, self.relevant_times = cross_Laplaican_eigenvalues_less_memory(self.f, device="cpu", Laplacian_fun=self.Laplacian_fun, min_dim=self.min_dim, max_dim=self.max_dim, compute_only_trace=self.compute_only_trace)
                # self.points = [[(p.birth, p.death) for p in self.dgms[q]] for q in range(self.max_dim+1)]
                plot_trace_diagram(self.f, self.eigenvalues, show=show, max_dim=self.max_dim, min_dim=self.min_dim, title="Laplacian diagram")
        else:
            print("No diagram to show." + "Please add filtration first."*(self.f is None) + "Please add Laplacian function first."*(self.Laplacian_fun is None))


    
    def evaluate_on_interval(self, n):
        x_axis = np.linspace(0, self.max_t, n)

        self.evaluations = {q: {k: [] for k in range(len(self.points[q]))} for q in range(self.min_dim, self.max_dim+1)}
        for t in x_axis:
            for q in range(self.min_dim, self.max_dim+1):
                evaluation_for_every_k = np.array([w/(q+2)*max(t-b, 0) if t<(b+d)/2 else w/(q+2)*max(d-t, 0) for b, d, w in self.points[q] if d != np.inf])
                evaluation_for_every_k[::-1].sort()
                for k in range(len(self.points[q])):
                    if k < len(evaluation_for_every_k):
                        self.evaluations[q][k].append(evaluation_for_every_k[k])
                    else:
                        self.evaluations[q][k].append(0)
        
        for q in range(self.min_dim, self.max_dim+1):
            for k in range(len(self.points[q])):
                self.evaluations[q][k] = np.array(self.evaluations[q][k])
        # self.evaluations = [np.array([self(t, p=dim) for t in x_axis]) for dim in range(self.max_dim+1)]
        return self.evaluations
    

        x_axis = np.linspace(0, self.max_t, n)
        self.evaluations = [np.array([self(t, q=dim) for t in x_axis]) for dim in range(self.min_dim, self.max_dim+1)]
        return self.evaluations

    def __call__(self, t, k, q=0):
        # Just multiplying by the weight
        # return np.sum([w*max(t-b, 0) if t<(b+d)/2 else w*max(d-t, 0) for b, d, w in self.points[q] if d != np.inf])

        # Normalizing the weight by the dimension
        return np.partition([w/(q+2)*max(t-b, 0) if t<(b+d)/2 else w/(q+2)*max(d-t, 0) for b, d, w in self.points[q] if d != np.inf], -(k+1))[-(k+1)]
        return np.sum([w/(q+2)*max(t-b, 0) if t<(b+d)/2 else w/(q+2)*max(d-t, 0) for b, d, w in self.points[q] if d != np.inf])
    
    def norm(self, return_error=False, limit=1000, use_evaluations=True, max_k=None, dim = None, sum_first = False):
        if use_evaluations:
            integral = 0
            if dim is None:
                q_bar = range(self.min_dim, self.max_dim+1)
            else:
                q_bar = [dim]
            for q in q_bar:
                if max_k is None:
                    k_bar = range(len(self.evaluations[q].keys()))
                else:
                    k_bar = range(min(max_k, len(self.evaluations[q].keys())))

                if sum_first:
                    sum_of_k = np.sum([self.evaluations[q][k] for k in k_bar], axis=0)
                    integral += integrate.trapezoid(sum_of_k**2, dx=self.max_t/sum_of_k.shape[0])
                else:
                    for k in k_bar:
                        integral += integrate.trapezoid(self.evaluations[q][k]**2, dx=self.max_t/len(self.evaluations[q][k]))
            return integral
        if self.computed_norm_limit < limit:
            self.computed_norm = 0
            self.norm_error = 0
            for q in range(self.max_dim+1):
                for k in range(len(self.evaluations[q].keys())):
                    integrate_result = integrate.quad(lambda t: self(t, k, q=q)**2, 0, self.max_t, limit=limit)
                    self.computed_norm += integrate_result[0]
                    self.norm_error += integrate_result[1]
        if return_error:
            return self.computed_norm, self.norm_error
        return self.computed_norm
    
    def sum_k(self):
        new_land = Lap_Landscape(max_t = self.max_t, min_dim = self.min_dim, max_dim=self.max_dim)
        new_land.evaluations = {q: {} for q in range(self.min_dim, self.max_dim+1)}
        for q in range(self.min_dim, self.max_dim+1):
            k_combined = [self.evaluations[q][k] for k in range(len(self.evaluations[q].keys()))]
            if len(k_combined) > 0:
                new_land.evaluations[q][0] = np.sum(k_combined, axis=0)
            elif q > self.min_dim:
                new_land.evaluations[q][0] = np.zeros(len(new_land.evaluations[q-1][0]))
            else:
                new_land.evaluations[q][0] = np.zeros(1000)
        return new_land

        
    # def compute_statistic(self, land_list, limit=1000):
    #     if isinstance(land_list, list):
    #         difference = 0
    #         error = 0
    #         sum_landscape = np.sum([land for land in land_list])
    #         n_others = len(land_list)

    #         for q in range(self.max_dim+1):
    #             integrate_result = integrate.quad(lambda t: (self(t, q=q) - 1/n_others*sum_landscape(t, q=q))**2, 0, max(self.max_t, *[land.max_t for land in land_list]), limit=limit)
    #             difference += integrate_result[0]
    #             error += integrate_result[1]
    #         print("Error: ", error)
    #         return difference
       
    def __add__(self, other):
        if self.max_t != other.max_t:
            raise ValueError("Landscapes must have the same max_t")
        
        new_land = Lap_Landscape(max_t = self.max_t, min_dim = min(self.min_dim, other.min_dim), max_dim=min(self.max_dim, other.max_dim))

        new_land.evaluations = {q: {k: self.evaluations[q][k] + other.evaluations[q][k] for k in range(min(len(other.evaluations[q].keys()), len(self.evaluations[q].keys())))} for q in range(new_land.min_dim, new_land.max_dim+1)}
        
        for q in range(new_land.min_dim, new_land.max_dim+1):
            if len(self.evaluations[q]) == 0:
                new_land.evaluations[q] = other.evaluations[q]
            elif len(other.evaluations[q]) == 0:
                new_land.evaluations[q] = self.evaluations[q]
            elif len(self.evaluations[q][0]) != len(other.evaluations[q][0]):
                raise ValueError("Landscapes must have the same number of evaluations")
            
            if len(self.evaluations[q].keys()) < len(other.evaluations[q].keys()):
                for k in range(len(self.evaluations[q].keys()), len(other.evaluations[q].keys())):
                    new_land.evaluations[q][k] = other.evaluations[q][k]
            elif len(self.evaluations[q].keys()) > len(other.evaluations[q].keys()):
                for k in range(len(other.evaluations[q].keys()), len(self.evaluations[q].keys())):
                    new_land.evaluations[q][k] = self.evaluations[q][k]
        # new_land.evaluations = [self.evaluations[p] + other.evaluations[p] for p in range(new_land.max_dim+1)]
        return new_land
    
    def __sub__(self, other):
        if self.max_t != other.max_t:
            raise ValueError("Landscapes must have the same max_t")
        
        new_land = Lap_Landscape(max_t = self.max_t, min_dim = min(self.min_dim, other.min_dim), max_dim=min(self.max_dim, other.max_dim))

        new_land.evaluations = {q: {k: self.evaluations[q][k] - other.evaluations[q][k] for k in range(min(len(other.evaluations[q].keys()), len(self.evaluations[q].keys())))} for q in range(new_land.min_dim, new_land.max_dim+1)}
        
        for q in range(new_land.min_dim, new_land.max_dim+1):
            if len(self.evaluations[q]) == 0:
                for k in range(len(other.evaluations[q].keys())):
                    new_land.evaluations[q][k] = -other.evaluations[q][k]
            elif len(other.evaluations[q]) == 0:
                for k in range(len(self.evaluations[q].keys())):
                    new_land.evaluations[q][k] = self.evaluations[q][k]
            elif len(self.evaluations[q][0]) != len(other.evaluations[q][0]):
                raise ValueError("Landscapes must have the same number of evaluations")
            
            if len(self.evaluations[q].keys()) < len(other.evaluations[q].keys()):
                for k in range(len(self.evaluations[q].keys()), len(other.evaluations[q].keys())):
                    new_land.evaluations[q][k] = -other.evaluations[q][k]
            elif len(self.evaluations[q].keys()) > len(other.evaluations[q].keys()):
                for k in range(len(other.evaluations[q].keys()), len(self.evaluations[q].keys())):
                    new_land.evaluations[q][k] = self.evaluations[q][k]

        # new_land.evaluations = [self.evaluations[p] - other.evaluations[p] for p in range(new_land.max_dim+1)]
        return new_land
    
    def __mult__(self, other):
        if np.isscalar(other):
            new_land = Lap_Landscape(max_t = self.max_t, min_dim = self.min_dim, max_dim=self.max_dim)
            new_land.evaluations = {q: {k: self.evaluations[q][k] * other for k in range(len(self.evaluations[q].keys()))} for q in range(self.min_dim, self.max_dim+1)}
            # new_land.evaluations = [self.evaluations[p] * other for p in range(self.max_dim+1)]
            return new_land
        else:
            raise ValueError("Multiplication is only supported with scalars")
    
    def __rmult__(self, other):
        if np.isscalar(other):
            new_land = Lap_Landscape(max_t = self.max_t, min_dim = self.min_dim, max_dim=self.max_dim)
            new_land.evaluations = {q: {k: self.evaluations[q][k] * other for k in range(len(self.evaluations[q].keys()))} for q in range(self.min_dim, self.max_dim+1)}
            # new_land.evaluations = [self.evaluations[p] * other for p in range(self.max_dim+1)]
            return new_land
        else:
            raise ValueError("Multiplication is only supported with scalars")
        
    def __truediv__(self, other):
        if isinstance(self, Lap_Landscape):
            if np.isscalar(other):
                new_land = Lap_Landscape(max_t = self.max_t, min_dim = self.min_dim, max_dim=self.max_dim)
                new_land.evaluations = {q: {k: self.evaluations[q][k] / other for k in range(len(self.evaluations[q].keys()))} for q in range(self.min_dim, self.max_dim+1)}
                return new_land
            else:
                print("other type:", type(other))
                raise ValueError("Division is only supported with scalars")
        elif isinstance(other, Lap_Landscape):
            if np.isscalar(self):
                new_land = Lap_Landscape(max_t = other.max_t, min_dim = other.min_dim, max_dim=other.max_dim)
                new_land.evaluations = {q: {k: other.evaluations[q][k] / self for k in range(len(other.evaluations[q].keys()))} for q in range(other.min_dim, other.max_dim+1)}
                return new_land
            else:
                raise ValueError("Division is only supported with scalars")
        
    def __rtruediv__(self, other):
        if np.isscalar(other):
            new_land = Lap_Landscape(max_t = other.max_t, min_dim = other.min_dim, max_dim=other.max_dim)
            new_land.evaluations = {q: {k: other.evaluations[q][k] / self for k in range(len(other.evaluations[q].keys()))} for q in range(other.min_dim, other.max_dim+1)}
            # new_land.evaluations = [other / self.evaluations[p] for p in range(self.max_dim+1)]
            return new_land
        else:
            raise ValueError("Division is only supported with scalars")


    def plot(self, ax=None, max_k=3, limits=None):
        if ax is None:
            fig, ax = plt.subplots(1, self.max_dim+1-self.min_dim, figsize=(5*(self.max_dim+1-self.min_dim), 5))
            
        for dim in range(self.min_dim, self.max_dim+1):
            if self.max_dim+1-self.min_dim > 1:
                cur_ax = ax[dim-self.min_dim]
            else:
                cur_ax = ax

            for k in range(min(len(self.evaluations[dim].keys()), max_k)):
                # max_non_zero = len(self.evaluations[dim])
                # for i in range(len(self.evaluations[dim])-1, -1, -1):
                #     if self.evaluations[dim][k][i] != 0:
                #         max_non_zero = i+1
                #         break
                # ax[dim].plot(np.linspace(0, self.max_t, len(self.evaluations[dim][k]))[:max_non_zero], self.evaluations[dim][k][:max_non_zero], c={0: "b", 1: "r", 2: "g", 3: "y"}[dim])
                cur_ax.plot(np.linspace(0, self.max_t, len(self.evaluations[dim][k])), self.evaluations[dim][k])#, c={0: "b", 1: "r", 2: "g", 3: "y"}[dim])
            
            if limits is not None:
                if isinstance(limits[0], list):
                    if len(limits) > dim-self.min_dim:
                        cur_ax.set_xlim(limits[dim-self.min_dim][0], limits[dim-self.min_dim][1])
                        cur_ax.set_ylim(limits[dim-self.min_dim][2], limits[dim-self.min_dim][3])
                else:
                    cur_ax.set_xlim(limits[0], limits[1])
                    cur_ax.set_ylim(limits[2], limits[3])
            cur_ax.legend([f"k={k}" for k in range(min(len(self.evaluations[dim].keys()), max_k))])
            cur_ax.set_xlabel("t")
            cur_ax.set_ylabel("Landscape")
            cur_ax.set_title(f"Lap Landscape dim {dim}")
        return ax

    # def plot(self, ax=None):
    #     if ax is None:
    #         fig, ax = plt.subplots(1, self.max_dim+1-self.min_dim, figsize=(5*(self.max_dim+1-self.min_dim), 5))
        
        

    #     for dim in range(self.min_dim, self.max_dim+1):
    #         if self.max_dim+1-self.min_dim > 1:
    #             cur_ax = ax[dim-self.min_dim]
    #         else:
    #             cur_ax = ax

    #         for i in range(len(self.evaluations[dim-self.min_dim])-1, -1, -1):
    #             if self.evaluations[dim-self.min_dim][i] != 0:
    #                 max_non_zero = i
    #                 break
    #         cur_ax.plot(np.linspace(0, self.max_t, len(self.evaluations[dim-self.min_dim]))[:max_non_zero], self.evaluations[dim-self.min_dim][:max_non_zero], c={0: "b", 1: "r", 2: "g", 3: "y"}[dim])
    #         cur_ax.set_xlabel("t")
    #         cur_ax.set_ylabel("Landscape")
    #         cur_ax.set_title(f"Landscape dim {dim}")
        
    #     # ax.legend([f"Dim {i}" for i in range(self.max_dim+1)])
        