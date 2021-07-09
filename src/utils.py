import math
import gzip, shutil
import plotly.graph_objects as go
from tqdm.autonotebook import tqdm
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch

def plot_pointcloud(verts: torch.Tensor, descr: Optional[str] = None, path: Optional[str] = None) -> None:
    """
    Plot a pointcloud.

    :param verts: vertices of the point cloud
    """
    x = verts[:,0]
    y = verts[:,1]
    z = verts[:,2]
    descr = "Pointcloud" if descr == None else descr
    layout = {"title": descr}
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                   mode='markers', marker=dict(size=3, opacity=0.8))], layout=layout)
    fig.update_layout(scene = dict(
                    xaxis = dict(
                        nticks = 4,
                        range=[-2,2],
                        backgroundcolor="rgb(230, 200, 200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),
                    yaxis = dict(
                        nticks = 4,
                        range=[-2,2],
                        backgroundcolor="rgb(200, 230, 200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    zaxis = dict(
                        nticks = 4,
                        range=[-2,2],
                        backgroundcolor="rgb(200, 200, 230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),),
                  )
    if path != None:
        fig.write_image(path)
    fig.show()

def plot_mesh(verts: torch.Tensor, faces: torch.Tensor, descr: Optional[str] = None, path: Optional[str] = None) -> None:
    """
    Plot a mesh.

    :param verts: vertices of the mesh
    :param faces: triangles of the mesh
    """
    x = verts[:,0]
    y = verts[:,1]
    z = verts[:,2]
    # i, j and k give the vertices of triangles
    i = faces[0]
    j = faces[1]
    k = faces[2]
    descr = "Mesh" if descr == None else descr
    layout = {"title": descr}
    fig = go.Figure(data=[go.Mesh3d(x = x, y = y, z = z, i = i, j = j, k = k)])

    fig.update_layout(scene = dict(
                    xaxis = dict(
                        nticks = 4,
                        range=[-2,2],
                        backgroundcolor="rgb(230, 200, 200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),
                    yaxis = dict(
                        nticks = 4,
                        range=[-2,2],
                        backgroundcolor="rgb(200, 230, 200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    zaxis = dict(
                        nticks = 4,
                        range=[-2,2],
                        backgroundcolor="rgb(200, 200, 230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),),
                  )

    if path != None:
        fig.write_image(path)
    fig.show()

def plot_scores_heatmap(scores: Dict, idx_to_transf: Dict, path: str) -> None:
    x = [str(idx_to_transf[idx]) for idx in scores.keys()]
    num_exps = len(x)
    y = [f'Expert {i}' for i in range(1, num_exps + 1)]

    z = [exps_scores for _, exps_scores in scores.items()]
    z = torch.stack(z).to('cpu')

    layout = {"title": "Winning experts on each transformations"}
    fig = go.Figure(data=go.Heatmap(x=x, y=y, z=z, colorscale="gray"), layout=layout)
    fig.write_image(path+"/scores_heatmap.png")
    fig.show()

def tranformations_distribution(dataset):
    transfs = {transf: 0 for transf, _ in dataset.transf_to_idx.items()}
    for data in tqdm(dataset, desc=f'Exploring dataset...', total=len(dataset)):
        transfs[dataset.idx_to_transf[data.transf.item()]] += 1
    print(transfs)

def Ry(angle, degrees=True):

    if degrees:
        
        cy = np.cos(np.deg2rad(angle))
        sy = np.sin(np.deg2rad(angle))
        
    else:
        
        cy = np.cos(angle)
        sy = np.sin(angle)
        
    Ry = np.array(
    [[cy , 0  , -sy, 0  ],
     [0  , 1  , 0  , 0  ],
     [sy , 0  , cy , 0  ],
     [0  , 0  , 0  , 1  ]])
    
    return Ry

def img_to_point_cloud(input_image):
    coords = np.transpose(np.nonzero(input_image))
    H, W = input_image.shape
    input_image = input_image / 255.0

    cloud = []
    faces = []
    for n in range(len(coords)):
        x = coords[n][0]
        y = coords[n][1]
        z = 0.1 * input_image[x][y]
        right = [x+1, y]
        diagonal = [x+1, y+1]
        down = [x, y+1]
        idx_right = np.argwhere(np.all(coords == right, axis=1))
        idx_diagonal = np.argwhere(np.all(coords == diagonal, axis=1))
        idx_down = np.argwhere(np.all(coords == down, axis=1))
        if len(idx_right) != 0 and len(idx_diagonal) != 0:
            face_1 = torch.LongTensor([n, idx_right.item(), idx_diagonal.item()])
            faces.append(face_1)
        if len(idx_down) != 0 and len(idx_diagonal) != 0:
            face_2 = torch.LongTensor([n, idx_down.item(), idx_diagonal.item()])
            faces.append(face_2)
        cloud.append([x/W, y/H, z])

    cloud = np.array(cloud)

    # rotate to vertical
    transf = np.c_[ cloud, np.ones(cloud.shape[0]) ]
    transf = transf @ Ry(90)
    cloud = transf[:,:-1]

    pos = torch.from_numpy(cloud).float()
    faces = torch.stack(faces) if len(faces) != 0 else None
    if faces != None:
        faces = faces.t().contiguous()

    return pos, faces

def extract_gz(file_in, file_out):
    with gzip.open(file_in, 'r') as f_in, open(file_out, 'wb') as f_out:
      shutil.copyfileobj(f_in, f_out)
