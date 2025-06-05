from genericpath import isfile
from PV_Functions import generate_cross_section
from Laplacian_Functions_torch import *
import networkx

from sklearn.cluster import DBSCAN

from tdamapper.learn import MapperAlgorithm
from tdamapper.cover import CubicalCover
from tdamapper.plot import MapperPlot

from tdamapper._plot_plotly import aggregate_graph
from landscapes import Landscape, Lap_Landscape

import pickle
import os

from PV_Functions import generate_cross_section_centers
import gudhi 

from pyballmapper import BallMapper
from pyballmapper.plotting import kmapper_visualize


def filtration_from_image_ballmapper(image, background_value = None, 
                          eps=None, save_to_html=None):
    """
    Create a filtration from an image by removing the background and applying Mapper.
    The image should be a PIL Image object.

    Parameters:
    image (PIL.Image): The input image to process.
    background_value: The value to consider as background (optional). If None, it will be calculated as the mean of the image.
    eps (float): The epsilon value for the BallMapper. If None, images_size/80 will be used.
    save_to_html (str): If provided, the resulting Mapper will be saved to an HTML file at this path.
    """

    # Convert the image to a grayscale numpy array
    im_gray = 1-np.array(image.convert("L")).astype(int)/255

    # Remove the background
    if background_value is None:
        background_value = np.mean(im_gray)
    
    if eps is None:
        eps = im_gray.shape[0] / 80

    X = np.vstack(np.where(im_gray>background_value)).T
    y = np.array([im_gray[X[i,0], X[i,1]] for i in range(X.shape[0])])

    # Create a Mapper cover and clustering
    bm = BallMapper(X=X.astype(np.float64), eps=eps) #coloring_df = pd.DataFrame(y, columns=["y"]))
    points_covered_by_landmarks = bm.points_covered_by_landmarks

    node_to_color = {node_i: np.mean([y[point_i] for point_i in points_covered_by_landmarks[node_i]]) for node_i in points_covered_by_landmarks.keys()}

    if save_to_html is not None:
        kmapper_visualize(
            bm, coloring_df=pd.DataFrame(y, columns=["y"]), path_html=save_to_html, title="BallMapper Filtration"
        )

    # Create a filtration from the graph
    simplices = []
    for c in networkx.enumerate_all_cliques(bm.Graph):
        # if len(c) <= 2:
        simplices.append((c, np.max([node_to_color[i] for i in c])))
    f = dio.Filtration()
    max_time = 0
    for vertices, time in simplices:
        f.append(dio.Simplex(vertices, time))
        max_time = max(max_time, time)
    # print(max_time)
    f.sort()

    return f

def filtration_from_image(image, background_value = None, 
                          cover=CubicalCover(n_intervals=15, overlap_frac=0.3, algorithm="standard"),
                          cluster_algorithm=DBSCAN(eps=5)):
    """
    Create a filtration from an image by removing the background and applying Mapper.
    The image should be a PIL Image object.

    Parameters:
    image (PIL.Image): The input image to process.
    background_value: The value to consider as background (optional). If None, it will be calculated as the mean of the image.
    """

    # Convert the image to a grayscale numpy array
    im_gray = 1-np.array(image.convert("L")).astype(int)/255

    # Remove the background
    if background_value is None:
        background_value = np.mean(im_gray)

    X = np.vstack(np.where(im_gray>background_value)).T
    y = np.array([im_gray[X[i,0], X[i,1]] for i in range(X.shape[0])])

    # Create a Mapper cover and clustering
    graph = MapperAlgorithm(cover, cluster_algorithm).fit_transform(X, X)
    node_col = aggregate_graph(y, graph, agg=np.nanmean)

    # Create a filtration from the graph
    simplices = []
    for c in networkx.enumerate_all_cliques(graph):
        # if len(c) <= 2:
        simplices.append((c, np.max([node_col[i] for i in c])))
    f = dio.Filtration()
    max_time = 0
    for vertices, time in simplices:
        f.append(dio.Simplex(vertices, time))
        max_time = max(max_time, time)
    # print(max_time)
    f.sort()

    return f

def poissonPointProcess(intensity, xMin=0, xMax=1, yMin=0, yMax=1, plot=False, seed=None):
    xDelta=xMax-xMin;yDelta=yMax-yMin #rectangle dimensions
    areaTotal=xDelta*yDelta
    if seed is not None:
        np.random.seed(seed) #set the random seed for reproducibility

    #Simulate a Poisson point process
    numbPoints = np.random.poisson(intensity*areaTotal);#Poisson number of points
    xx = xDelta*np.random.uniform(0,1,numbPoints)+xMin;#x coordinates of Poisson points
    yy = yDelta*np.random.uniform(0,1,numbPoints)+yMin;#y coordinates of Poisson points

    #Plot the points
    if plot:
        plt.figure()
        plt.scatter(xx,yy,s=2)
        plt.xlim(xMin,xMax)
        plt.ylim(yMin,yMax)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Poisson Point Process, n={numbPoints}')
        plt.show()
    return xx,yy

def PointProcessFiltration(n, xMin=0, xMax=1, yMin=0, yMax=1, plot=False, seed=None, max_r=None, method = "poisson"):
    if max_r is None:
        max_r = max(xMax-xMin, yMax-yMin)
    
    # xx, yy = poissonPointProcess(intensity, xMin=xMin, xMax=xMax, yMin=yMin, yMax = yMax, seed=seed)
    if method == "poisson":
        points = generate_cross_section_centers(n, generation_seed = seed, generation_method = "PV", cross_section_locations=[0.5])[0.5]
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random
        xx = rng.uniform(xMin,xMax,points.shape[0]);#x coordinates of Poisson points
        yy = rng.uniform(yMin,yMax,points.shape[0]);#y coordinates of Poisson points
    else:
        points = generate_cross_section_centers(n, generation_seed = seed, generation_method = method, cross_section_locations=[0.5])[0.5]
        xx = points[:, 0]
        yy = points[:, 1]
    numbPoints = len(xx)
    # print(f"Number of points: {numbPoints}")

    if plot:
        plt.figure()
        plt.scatter(xx,yy,s=2)
        plt.xlim(xMin,xMax)
        plt.ylim(yMin,yMax)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{method} Process, n={numbPoints}')
        plt.show()

    # f = dio.fill_rips(np.array([xx, yy]).T, 1, max_r)
    f = dio.Filtration()
    tree = gudhi.alpha_complex.AlphaComplex(points=np.array([xx, yy]).T).create_simplex_tree(max_alpha_square=max_r)
    for vertices, t in tree.get_filtration():
        f.append(dio.Simplex(vertices, t))
    f.sort()
    return f

def Laplacian_fun(B22_st, B22_stm1, B22_sm1t, B22_sm1tm1, eye):
    # return B22_stm1@B22_sm1t@(eye-B22_st)@B22_sm1t
    # return B22_sm1t@(eye-B22_st)@B22_sm1t@B22_stm1
    # return B22_sm1t@B22_stm1@(eye-B22_st)@B22_stm1
    # return B22_stm1@(eye-B22_st)@B22_stm1@B22_sm1t
    # return B22_sm1tm1-B22_sm1t-B22_stm1+B22_st

    
    # return B22_stm1@B22_sm1t+B22_sm1t@B22_stm1-2*B22_st
    # return eye-B22_st

    # return B22_stm1@(eye-B22_st)@(eye - B22_sm1tm1 + B22_sm1t)

    # Best
    return B22_sm1t@B22_stm1-B22_st

def main():
    max_r = 1
    ball_eps = 15
    color_seed = 124

    for edge_color in ["random"]:
        for method in ["PV", "cluster", "HC"]:
            seed_bar = tqdm(range(100))

            if edge_color == "random":
                save_location = f"../ballmapper_landscapes/eps_{ball_eps}/{edge_color}_{color_seed}/{method}"
            else:
                save_location = f"../ballmapper_landscapes/eps_{ball_eps}/{edge_color}/{method}"

            for seed in seed_bar:
                try:
                    seed_bar.set_description(f"Method: {method}, Seed: {seed}")

                    if os.path.isfile(os.path.join(save_location, f"normal_{seed}.pkl")) and \
                       os.path.isfile(os.path.join(save_location, f"laplacian_{seed}.pkl")):
                        # print(f"Files for method {method} with seed {seed} already exist. Skipping...")
                        continue

                    images = generate_cross_section(250, save_images=True, generation_method=method, 
                                                    length_of_cube_sides= 1,
                                                    sample_params={},
                                                    cross_section_locations=np.array([0.25]), 
                                                    generation_seed=seed,
                                                    color_seed=color_seed, 
                                                    edge_colors=edge_color)

                    # f = filtration_from_image(images[0],
                    #                         cover=CubicalCover(n_intervals=25, overlap_frac=0.3, algorithm="standard"))
                    f = filtration_from_image_ballmapper(images[0], eps=ball_eps)

                    land = Landscape(f, show_diagram=False, max_t=max_r)
                    # land.show_diagram(show=False)
                    # im_name = f"{method}_{seed}"
                    # plt.savefig("../figures/small_tests_Mapper_gradient/" + im_name + "_normal_diagram.png")
                    # land.plot()
                    # plt.savefig("../figures/small_tests_Mapper_gradient/" + im_name + "_normal_landscape.png")

                    lap_land = Lap_Landscape(f, show_trace_diagram=False, min_dim=0, max_dim = 1, Laplacian_fun=Laplacian_fun,compute_only_trace=True, max_t=max_r)
                    # lap_land.show_trace_diagram(show=False)
                    # plt.savefig("../figures/small_tests_Mapper_gradient/" + im_name + "_lap_diagram.png")
                    # lap_land.plot()
                    # plt.savefig("../figures/small_tests_Mapper_gradient/" + im_name + "_lap_landscape.png")
                    plt.close("all")

                    
                    os.makedirs(save_location, exist_ok=True)
                    with open(os.path.join(save_location, f"normal_{seed}.pkl"), "wb") as f:
                        land.f = None
                        pickle.dump(land, f)

                    with open(os.path.join(save_location, f"laplacian_{seed}.pkl"), "wb") as f:
                        lap_land.f = None
                        pickle.dump(lap_land, f)
                except KeyboardInterrupt:
                    raise KeyboardInterrupt("Process interrupted by user.")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print(f"Error processing method {method} with seed {seed}. Skipping...")
                    continue

if __name__ == "__main__":
    main()