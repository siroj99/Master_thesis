from genericpath import isfile

from numpy import cross
from scipy import cluster
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

class generation_args():
    max_r = 1
    
    # Saving parameters
    path_to_output = "../ballmapper_landscapes_normal_function"
    # path_to_output = "../alpha_landscapes"
    redo_landscapes = False

    # Edge colors
    edge_colors = ["random", "gradient"]
    # edge_colors = ["gradient"]
    # edge_colors = ["alpha"]
    color_seed = 124

    # Generation parameters
    n_points = 250
    methods = ["PV", "cluster", "HC"]
    seeds = range(100)
    cross_section_location = 0.25
    
    use_new_filtration = False

    # BallMapper parameters
    use_ballmapper = True
    ball_eps = 15

    # Normal Mapper parameters
    cover = CubicalCover(n_intervals=25, overlap_frac=0.3, algorithm="standard")
    cluster_algorithm = DBSCAN(eps=5)

def graph_filtration(graph, node_to_color):
    simplices = []
    # points_covered_by_landmarks = bm.points_covered_by_landmarks
    # print(points_covered_by_landmarks)
    # # print(reversed(list(networkx.enumerate_all_cliques(bm.Graph))))
    # node_to_color = {node_i: np.mean([y_1[point_i] for point_i in points_covered_by_landmarks[node_i]]) for node_i in points_covered_by_landmarks.keys()}
    nodes_done = set()
    simplex_to_nodes = {}
    nodes_to_simplex = {node_i: node_i for node_i in graph.nodes}
    for c in reversed(list(networkx.enumerate_all_cliques(graph))):
        if any([i in nodes_done for i in c]) and len(c) > 2:
            continue
        elif len(c) > 2:
            nodes_done.update(c)

            for i in c:
                nodes_to_simplex[i] = min(c)
            simplices.append(([min(c)], np.mean([node_to_color[i] for i in c])))
            simplex_to_nodes[min(c)] = c
        elif len(c) == 2:
            if c[0] not in nodes_done:
                simplex_to_nodes[c[0]] = [c[0]]
                simplices.append(([c[0]], node_to_color[c[0]]))
                nodes_done.add(c[0])
            if c[1] not in nodes_done:
                simplex_to_nodes[c[1]] = [c[1]]
                simplices.append(([c[1]], node_to_color[c[1]]))
                nodes_done.add(c[1])
            if nodes_to_simplex[c[0]] != nodes_to_simplex[c[1]]:
                avg_0 = np.mean([node_to_color[i] for i in simplex_to_nodes[nodes_to_simplex[c[0]]]])
                avg_1 = np.mean([node_to_color[i] for i in simplex_to_nodes[nodes_to_simplex[c[1]]]])
                simplices.append(([nodes_to_simplex[c[0]], nodes_to_simplex[c[1]]], np.max([avg_0, avg_1])))

    # for s in simplex_to_nodes.keys():
    #     simplices.append([s], np.mean([node_to_color[i] for i in simplex_to_nodes[s]]))

        # if len(c) <= 2:
        # simplices.append((c, np.max([node_to_color[i] for i in c])))

    f = dio.Filtration()
    max_time = 0
    for vertices, time in simplices:
        f.append(dio.Simplex(vertices, time))
        max_time = max(max_time, time)
    f.sort()

    return f, simplex_to_nodes, nodes_to_simplex

def filtration_from_image_ballmapper(image, background_value = None, use_new_filtration=False,
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
    if use_new_filtration:
        f, simplex_to_nodes, nodes_to_simplex = graph_filtration(bm.Graph, node_to_color)
    else:
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

def filtration_from_image(image, background_value = None, use_new_filtration=False,
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
    if use_new_filtration:
        f, simplex_to_nodes, nodes_to_simplex = graph_filtration(graph, node_col)
    else:
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

def PointProcessFiltration(n, xMin=0, xMax=1, yMin=0, yMax=1, plot=False, seed=None, max_r=None, method = "poisson", cross_section_location=0.5):
    if max_r is None:
        max_r = max(xMax-xMin, yMax-yMin)
    
    # xx, yy = poissonPointProcess(intensity, xMin=xMin, xMax=xMax, yMin=yMin, yMax = yMax, seed=seed)
    if method == "poisson":
        points = generate_cross_section_centers(n, generation_seed = seed, generation_method = "PV", cross_section_locations=[cross_section_location])[cross_section_location]
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random
        xx = rng.uniform(xMin,xMax,points.shape[0]);#x coordinates of Poisson points
        yy = rng.uniform(yMin,yMax,points.shape[0]);#y coordinates of Poisson points
    else:
        points = generate_cross_section_centers(n, generation_seed = seed, generation_method = method, cross_section_locations=[cross_section_location])[cross_section_location]
        xx = points[:, 0]
        yy = points[:, 1]
    numbPoints = len(xx)
    # print(f"Number of points: {numbPoints}")

    if plot:
        plt.figure(figsize=(5,5))
        plt.scatter(xx,yy,s=5)
        plt.xlim(xMin,xMax)
        plt.ylim(yMin,yMax)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{method} Process')
        # plt.show()

    # f = dio.fill_rips(np.array([xx, yy]).T, 1, max_r)
    f = dio.Filtration()
    tree = gudhi.alpha_complex.AlphaComplex(points=np.array([xx, yy]).T).create_simplex_tree(max_alpha_square=(2*np.pi*max_r)**2)
    for vertices, t in tree.get_filtration():
        f.append(dio.Simplex(vertices, np.sqrt(t)/(2*np.pi))) # take square root and divide by 2pi to get the radius of the circle
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
    # return B22_sm1t@B22_stm1-B22_st

    # Normal
    return B22_stm1 - B22_st - B22_sm1tm1 + B22_sm1t

def main(args: generation_args):
    for edge_color in args.edge_colors:
        for method in args.methods:
            seed_bar = tqdm(args.seeds)
            if edge_color == "random":
                save_location = f"{args.path_to_output}/eps_{args.ball_eps}/{edge_color}_{args.color_seed}/{method}"
            else:
                save_location = f"{args.path_to_output}/eps_{args.ball_eps}/{edge_color}/{method}"

            for seed in seed_bar:
                seed_bar.set_description(f"Method: {method}, Seed: {seed}")
                try:
                    if os.path.isfile(os.path.join(save_location, f"normal_{seed}.pkl")) and \
                       os.path.isfile(os.path.join(save_location, f"laplacian_{seed}.pkl")) and not args.redo_landscapes:
                        # print(f"Files for method {method} with seed {seed} already exist. Skipping...")
                        continue
                    
                    if edge_color == "alpha":
                        f = PointProcessFiltration(args.n_points, plot=False, seed=seed, method=method, 
                                                   max_r=args.max_r, 
                                                   cross_section_location=args.cross_section_location)
                    
                    else:
                        images = generate_cross_section(args.n_points, save_images=True, generation_method=method, 
                                                        length_of_cube_sides= 1,
                                                        sample_params={},
                                                        cross_section_locations=np.array([args.cross_section_location]), 
                                                        generation_seed=seed,
                                                        color_seed=args.color_seed, 
                                                        edge_colors=edge_color)

                        if args.use_ballmapper:
                            f = filtration_from_image_ballmapper(images[0], use_new_filtration=args.use_new_filtration,
                                                                 eps=args.ball_eps)
                        else:
                            f = filtration_from_image(images[0], use_new_filtration=args.use_new_filtration,
                                                      cover=args.cover, 
                                                      cluster_algorithm = args.cluster_algorithm)
                        

                    land = Landscape(f, show_diagram=False, max_t=args.max_r)
                    # land.show_diagram(show=False)
                    # im_name = f"{method}_{seed}"
                    # plt.savefig("../figures/small_tests_Mapper_gradient/" + im_name + "_normal_diagram.png")
                    # land.plot()
                    # plt.savefig("../figures/small_tests_Mapper_gradient/" + im_name + "_normal_landscape.png")

                    lap_land = Lap_Landscape(f, show_trace_diagram=False, min_dim=0, max_dim = 1, Laplacian_fun=Laplacian_fun,compute_only_trace=True, max_t=args.max_r)
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
    args = generation_args()
    main(args)

    # args.ball_eps = 20
    # main(args)
    
    # args.ball_eps = 10
    # main(args)