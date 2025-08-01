import numpy as np
from shapely import length
import vorostereology as vs
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import pyvoro
import pandas as pd

from PIL import Image, ImageChops
import os
import os.path as osp

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    
def generate_points_PV(n, domain, rng=np.random):
    """
    NOTE: Only works for cubical domain
    """
    return rng.uniform(domain[0][0], domain[0][1],size=(n,3))

def generate_points_cluster(n, domain, rng=np.random, error_in_samples = 0.1,
                            n_parents = None, radiusCluster = None):
    xMin = domain[0][0]
    xMax = domain[0][1]
    yMin = domain[1][0]
    yMax = domain[1][1]
    zMin = domain[2][0]
    zMax = domain[2][1]

    if radiusCluster is None:
        radiusCluster = (xMax - xMin)/5
    # Parameters for the parent and daughter point processes
    # lambdaParent = 10;  # density of parent Poisson point process
    # lambdaDaughter = 100;  # mean number of points in each cluster
    # radiusCluster = 0.1;  # radius of cluster disk (for daughter points)

    # Extended simulation windows parameters
    rExt = radiusCluster;  # extension parameter -- use cluster radius
    xMinExt = xMin - rExt;
    xMaxExt = xMax + rExt;
    yMinExt = yMin - rExt;
    yMaxExt = yMax + rExt;
    zMinExt = zMin - rExt;
    zMaxExt = zMax + rExt;
    # rectangle dimensions
    xDeltaExt = xMaxExt - xMinExt;
    yDeltaExt = yMaxExt - yMinExt;
    zDeltaExt = zMaxExt - zMinExt;
    volumeTotalExt = xDeltaExt * yDeltaExt*zDeltaExt;  # volume of extended rectangle
    n_ori = n
    # Adjust n based on the error margin
    n = int(volumeTotalExt/((xMax-xMin)*(yMax-yMin)*(zMax-zMin))*n )

    # Simulate Poisson point process for the parents
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! numbPointsParent = rng.poisson(volumeTotalExt * lambdaParent);  # Poisson number of points !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if n_parents is not None:
        numbPointsParent = n_parents
    else:
        numbPointsParent = int(0.01*n)
    
    # x and y coordinates of Poisson points for the parent
    xxParent = xMinExt + xDeltaExt * rng.uniform(0, 1, numbPointsParent);
    yyParent = yMinExt + yDeltaExt * rng.uniform(0, 1, numbPointsParent);
    zzParent = zMinExt + zDeltaExt * rng.uniform(0, 1, numbPointsParent);
    # Simulate Poisson point process for the daughters (ie final poiint process)
    # np.random.poisson()
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! numbPointsDaughter = rng.poisson(lambdaDaughter, numbPointsParent); !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # numbPointsDaughter = np.array([n//n_parents]*n_parents)
    numbPointsDaughter = np.array([n//numbPointsParent]*numbPointsParent)
    
    numbPoints = sum(numbPointsDaughter);  # total number of point

    # Generate the (relative) locations in polar coordinates by
    # simulating independent variables.
    theta = 2 * np.pi * rng.uniform(0, 1, numbPoints);  # angular coordinates
    rho = radiusCluster * np.sqrt(rng.uniform(0, 1, numbPoints));  # radial coordinates
    V=2 * rng.uniform(0, 1, numbPoints)-1;

    # Convert from polar to Cartesian coordinates
    xx0 = rho * np.cos(theta)* np.sqrt(1-V**2);
    yy0 = rho * np.sin(theta)* np.sqrt(1-V**2);
    zz0 = rho * V
    # replicate parent points (ie centres of disks/clusters)
    xx = np.repeat(xxParent, numbPointsDaughter);
    yy = np.repeat(yyParent, numbPointsDaughter);
    zz = np.repeat(zzParent, numbPointsDaughter);
    # translate points (ie parents points are the centres of cluster disks)
    xx = xx + xx0;
    yy = yy + yy0;
    zz = zz + zz0;
    # thin points if outside the simulation window
    booleInside = ((xx >= xMin) & (xx <= xMax) & (yy >= yMin) & (yy <= yMax)& (zz >= zMin) & (zz <= zMax));
    # retain points inside simulation window
    xx = xx[booleInside];  
    yy = yy[booleInside];
    zz = zz[booleInside]

    s = np.array([xx.T,yy.T,zz.T]).T
    if (1-error_in_samples)*n_ori < s.shape[0] < (1+error_in_samples)*n_ori:
        # print(f"Generated {s.shape[0]} points, which is within the error margin of {error_in_samples*100:.2f}% of the target {n_ori} points.")
        return s
    # print(f"Generated {s.shape[0]} points, which is outside the error margin of {error_in_samples*100:.2f}% of the target {n_ori} points. Regenerating...")
    return generate_points_cluster(n_ori, domain, rng=rng, error_in_samples=error_in_samples,
                                   n_parents=n_parents, radiusCluster=radiusCluster)

def generate_points_HC(n, domain, rng=np.random,
                       radiusCore = None, error_in_samples = 0.1):
    xMin = domain[0][0]
    xMax = domain[0][1]
    yMin = domain[1][0]
    yMax = domain[1][1]
    zMin = domain[2][0]
    zMax = domain[2][1]

    if radiusCore is None:
        # For a cube of side length 1, this gives around 1000 points
        radiusCore = 0.017

    #Parameters for the parent and daughter point processes
    # lambdaPoisson=2000;#density of underlying Poisson point process
    # radiusCore=0.055;#radius of hard core

    #Extended simulation windows parameters
    rExt=radiusCore; #extension parameter -- use core radius
    xMinExt=xMin-rExt;
    xMaxExt=xMax+rExt;
    yMinExt=yMin-rExt;
    yMaxExt=yMax+rExt;
    zMinExt=zMin-rExt;
    zMaxExt=zMax+rExt;
    #rectangle dimensions
    xDeltaExt=xMaxExt-xMinExt;
    yDeltaExt=yMaxExt-yMinExt;
    zDeltaExt=zMaxExt-zMinExt;
    volumeTotalExt=xDeltaExt*yDeltaExt*zDeltaExt; #volume of extended rectangle

    #Simulate Poisson point process for the parents
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! numbPointsExt= np.random.poisson(volumeTotalExt*lambdaPoisson);#Poisson number !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    numbPointsExt = 5*n

    #x and y and z coordinates of Poisson points for the parent
    xxPoissonExt=xMinExt+xDeltaExt*rng.random(numbPointsExt);
    yyPoissonExt=yMinExt+yDeltaExt*rng.random(numbPointsExt);
    zzPoissonExt=zMinExt+zDeltaExt*rng.random(numbPointsExt);
        
    #thin points if outside the simulation window
    booleWindow=((xxPoissonExt>=xMin)&(xxPoissonExt<=xMax)&(yyPoissonExt>=yMin)&(yyPoissonExt<=yMax)&(zzPoissonExt>=zMin)&(zzPoissonExt<=zMax));
    indexWindow=np.arange(numbPointsExt)[booleWindow];
    #retain points inside simulation window
    xxPoisson=xxPoissonExt[booleWindow];
    yyPoisson=yyPoissonExt[booleWindow];
    zzPoisson=zzPoissonExt[booleWindow];
        
    numbPoints=len(xxPoisson); #number of Poisson points in window
    #create random marks for ages
    markAge=rng.random(numbPointsExt);
        
    ###START Removing/thinning points START###
    booleRemoveI=np.zeros(numbPoints, dtype=bool);#Index for removing points -- Matern I
    booleKeepII=np.zeros(numbPoints,dtype=bool);#Index for keeping points -- Matern II
    for ii in range(numbPoints):
        distTemp=np.hypot(xxPoisson[ii]-xxPoissonExt,yyPoisson[ii]-yyPoissonExt,zzPoisson[ii]-zzPoissonExt);  #distances to other points        
        booleInDisk=(distTemp<radiusCore)&(distTemp>0); #check if inside disk
            
            #Matern I
        booleRemoveI[ii]=any(booleInDisk);
            
            #Matern II
            #keep the younger points
        if len(markAge[booleInDisk])==0:
            booleKeepII[ii]=True;
                #Note: if markAge(booleInDisk) is empty, keepBooleII[ii]=True.
        else:
            booleKeepII[ii]=all(markAge[indexWindow[ii]]<markAge[booleInDisk]);
                
                
        ###END Removing/thinning points END###
        
        #Remove/keep points to generate Matern hard-core processes
        #Matérn I
    booleKeepI=~(booleRemoveI);
    xxMaternI=xxPoisson[booleKeepI];
    yyMaternI=yyPoisson[booleKeepI];
    zzMaternI=zzPoisson[booleKeepI];
        #Matérn II
    xxMaternII=xxPoisson[booleKeepII];
    yyMaternII=yyPoisson[booleKeepII];
    zzMaternII=zzPoisson[booleKeepII];

    s=np.array([xxMaternII.T,yyMaternII.T,zzMaternII.T]).T
    # sp=np.array([xxPoisson.T,yyPoisson.T,zzPoisson.T]).T

    if (1-error_in_samples)*n < s.shape[0] < (1+error_in_samples)*n:
        # print(f"Generated {s.shape[0]} points, which is within the error margin of {error_in_samples*100:.2f}% of the target {n} points.")
        print(f"Generated {s.shape[0]} points, which is within the error margin of {error_in_samples*100:.2f}% of the target {n} points.")
        print(f"radius core: {radiusCore}")
        return s
    elif s.shape[0] < (1-error_in_samples)*n:
        # print(f"Generated {s.shape[0]} points, which is below the error margin of {error_in_samples*100:.2f}% of the target {n} points. Regenerating with 25% smaller radius...")
        return generate_points_HC(n, domain, rng=rng,
                       radiusCore = 0.75*radiusCore, error_in_samples = error_in_samples)
    elif s.shape[0] > (1+error_in_samples)*n:
            
        # print(f"Generated {s.shape[0]} points, which is above the error margin of {error_in_samples*100:.2f}% of the target {n} points. Regenerating with 25% larger radius...")
        return generate_points_HC(n, domain, rng=rng,
                        radiusCore = 1.25*radiusCore, error_in_samples = error_in_samples)


def generate_cross_section(n, length_of_cube_sides = 1, generation_seed = None, color_seed = None, weights = None,
                           generation_method = "PV",
                           coefs_cross_section = np.array([1, 0.0, 0.0]),
                           cross_section_locations = np.arange(0.25, 0.7, 0.05),
                           plot_3d_visualization = False,
                           edge_colors = "random",
                           edge_value_limits = (0.3,1),
                           save_images = True,
                           save_location = "../figures/",
                           sample_params = {}):
    """
    Pictures are saved under: {save_location}/{generation_method}/n={n}_colors={edge_colors}_seed={seed}_loc={cross_section_locations[section_i]:.2f}.png

    sample_params: For generation_method="cluster": {"n_parents": 0.01*volume_ext_cube/volume_cube*n, "radiusCluster": length_of_cube_sides/5}
    """

    if generation_seed is None:
        rng_generation = np.random
    else:
        rng_generation = np.random.default_rng(seed=generation_seed)

    if color_seed is None:
        rng_color = np.random
    else:
        rng_color = np.random.default_rng(seed=color_seed)
    # Setting the variables
    if weights is None:
        weights = np.zeros(n)

    domain=[[0.0, length_of_cube_sides], [0.0, length_of_cube_sides], [0.0, length_of_cube_sides]]
    
    if generation_method == "PV":
        s = generate_points_PV(n, domain, rng=rng_generation)
    elif generation_method == "cluster":
        s = generate_points_cluster(n, domain, rng=rng_generation, **sample_params)
    elif generation_method == "HC":
        s = generate_points_HC(n, domain, rng=rng_generation, **sample_params)
    # print(f"Sampled {s.shape[0]} points.")

    # Computing 3d PV
    pycells=pyvoro.compute_voronoi(
        s, # point positions
        domain, # limits
        length_of_cube_sides, # block size
        # particle radii -- optional, and keyword-compatible arg.
        )
    
    # Setting cross sections
    cross_section_tot = []
    for x in range(len(cross_section_locations)):
        cross_section_tot += [vs.compute_cross_section(coeffs=coefs_cross_section, offset = np.array([cross_section_locations[x], 0, length_of_cube_sides]), points = s, domain=domain, weights = weights)]
    
    # Plotting 3d PV with cross sections highlighted
    if plot_3d_visualization:
        fig = plt.figure()
        # ax = Axes3D(fig)
        ax = fig.add_subplot(111, projection='3d')

        for cell_idx, cell in enumerate(pycells):
            for facet_idx, facet in enumerate(cell['faces']):
                idx = np.array(facet['vertices'])
                polygon = m3.art3d.Poly3DCollection([np.array(cell['vertices'])[idx]])
                polygon.set_edgecolor('k')
                polygon.set_alpha(0.1)
                ax.add_collection3d(polygon)

        for x in range(9):
            for cell in cross_section_tot[x]['3d']:
                polygon = m3.art3d.Poly3DCollection([cell])
                polygon.set_color("red")
                polygon.set_edgecolor('k')
                ax.add_collection3d(polygon)

            
        ax.set_xlim3d(domain[0])
        ax.set_ylim3d(domain[1])
        ax.set_zlim3d(domain[2])
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        plt.show(block=False)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)


    if save_images:
        os.makedirs(osp.join(save_location,generation_method), exist_ok=True)
    
    image_list = []
    all_indices = set()
    for cross_section in cross_section_tot:
        all_indices.update(cross_section["original_indices"])
    if edge_colors == "random":
        index_to_color = {i: (1, 1-x, 1-x) for i, x in zip(all_indices, rng_color.uniform(edge_value_limits[0], edge_value_limits[1], size=len(all_indices)))}
        
    for section_i in range(len(cross_section_tot)):
        if edge_colors == "completely-random":
            edgecolors = [(1, 1-c, 1-c) for c in rng_color.uniform(edge_value_limits[0], edge_value_limits[1], size=len(cross_section_tot[section_i]['2d']))]
        elif edge_colors == "random":
            edgecolors = [index_to_color[i] for i in cross_section_tot[section_i]['original_indices']]
        elif edge_colors == "gradient":
            edgecolors = [(1, 1-c, 1-c) for c in np.linspace(edge_value_limits[0], edge_value_limits[1], num=len(cross_section_tot[section_i]['2d']))]
        else:
            edgecolors = "red"
        coll = PolyCollection(cross_section_tot[section_i]['2d'], facecolors="white", edgecolors=edgecolors, closed=True)
        ax2.add_collection(coll)
        ax2.axis("equal")
        ax2.set_axis_off()
        fig2.tight_layout()
        plt.savefig("temp.png")
        im = Image.frombytes("RGB", fig2.canvas.get_width_height(), fig2.canvas.tostring_rgb())
        im = trim(im)
        crop_by = 5
        im = im.crop((crop_by, crop_by, im.width-crop_by, im.height-crop_by))

        # if generation_seed is None:

        #     im_name = f"n={n}_colors={edge_colors}_seed=random_loc={cross_section_locations[section_i]:.2f}.png"
        # else:
        if color_seed is None:
            im_name = f"n={n}_colors={edge_colors}{'-r' if edge_colors in ['random', 'completely-random'] else ''}_seed={generation_seed if generation_seed is not None else 'random'}_loc={cross_section_locations[section_i]:.2f}.png"
        else:
            im_name = f"n={n}_colors={edge_colors}{'-'+str(color_seed) if edge_colors in ['random', 'completely-random'] else ''}_seed={generation_seed if generation_seed is not None else 'random'}_loc={cross_section_locations[section_i]:.2f}.png"
        
        image_list.append(im)
        if save_images:
            im.save(osp.join(save_location,generation_method, im_name))

    os.remove("temp.png")
    return image_list

def generate_cross_section_centers(n, length_of_cube_sides = 1, generation_seed = None,  weights = None,
                           generation_method = "PV",
                           coefs_cross_section = np.array([1, 0.0, 0.0]),
                           cross_section_locations = np.arange(0.25, 0.7, 0.05),
                           sample_params = {}):
    """
    Pictures are saved under: {save_location}/{generation_method}/n={n}_colors={edge_colors}_seed={seed}_loc={cross_section_locations[section_i]:.2f}.png

    sample_params: For generation_method="cluster": {"n_parents": 0.01*volume_ext_cube/volume_cube*n, "radiusCluster": length_of_cube_sides/5}
    """

    if generation_seed is None:
        rng_generation = np.random
    else:
        rng_generation = np.random.default_rng(seed=generation_seed)

    # Setting the variables
    if weights is None:
        weights = np.zeros(n)

    domain=[[0.0, length_of_cube_sides], [0.0, length_of_cube_sides], [0.0, length_of_cube_sides]]
    
    if generation_method == "PV":
        s = generate_points_PV(n, domain, rng=rng_generation)
    elif generation_method == "cluster":
        s = generate_points_cluster(n, domain, rng=rng_generation, **sample_params)
    elif generation_method == "HC":
        s = generate_points_HC(n, domain, rng=rng_generation, **sample_params)
    else:
        print(f"Unknown generation method: {generation_method}.")
        return
    # print(f"Sampled {s.shape[0]} points.")

    # Setting cross sections
    cross_section_tot = []
    for x in range(len(cross_section_locations)):
        cross_section_tot += [vs.compute_cross_section(coeffs=coefs_cross_section, offset = np.array([cross_section_locations[x], 0, length_of_cube_sides]), points = s, domain=domain, weights = weights)]
    
    points_dict = {}

    for i_section, cross_section in enumerate(cross_section_tot):
        centroid0=[]
        for i in range(len( cross_section['2d'])):
            x = [p[0] for p in cross_section['2d'][i]]
            y = [p[1] for p in cross_section['2d'][i]]
            centroid0.append((sum(x) / len(cross_section['2d'][i]), sum(y) / len(cross_section['2d'][i])))
        points_dict[cross_section_locations[i_section]] = np.array(centroid0)
    # id_sec=pd.DataFrame([cross_section_locations[0]]*len(cross_section_tot[0]['2d']),columns=['id_sec'])
    # id_grains=pd.DataFrame(cross_section_tot[0]['original_indices'],columns=['id_grain'])
    # df0=pd.DataFrame(centroid0,columns=['x','y'])
    # df0=pd.concat([df0.reset_index(drop=True), id_sec, id_grains], axis=1)
    # for j in range(1,len(cross_section_tot)):
    #     x = [p[0] for p in cross_section_tot[j]['2d'][0]]
    #     y = [p[1] for p in cross_section_tot[j]['2d'][0]]
    #     centroidj=[(sum(x) / len(cross_section_tot[j]['2d'][0]), sum(y) / len(cross_section_tot[j]['2d'][0]))]
    #     for i in range(1,len(cross_section_tot[j]['2d'])):
    #         x = [p[0] for p in cross_section_tot[j]['2d'][i]]
    #         y = [p[1] for p in cross_section_tot[j]['2d'][i]]   
    #         centroidj.append((sum(x) / len(cross_section_tot[j]['2d'][i]), sum(y) / len(cross_section_tot[j]['2d'][i])))
    #     id_sec=pd.DataFrame([cross_section_locations[j]]*len(cross_section_tot[j]['2d']),columns=['id_sec'])
    #     id_grains=pd.DataFrame(cross_section_tot[j]['original_indices'],columns=['id_grain'])
    #     dfj=pd.DataFrame(centroidj,columns=['x','y'])
    #     dfj=pd.concat([dfj.reset_index(drop=True), id_sec, id_grains], axis=1)
    #     dfj_tot=pd.concat([df0,dfj])
    #     df0=dfj_tot 
    return points_dict  
