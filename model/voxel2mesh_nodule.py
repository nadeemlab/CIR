import torch.nn as nn
import torch 
import torch.nn.functional as F 

from pytorch3d.structures import Meshes 
from pytorch3d.ops import sample_points_from_meshes, SubdivideMeshes
from pytorch3d.loss import (chamfer_distance,  mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency)

import numpy as np
from itertools import product, combinations, chain
from scipy.spatial import ConvexHull

from IPython import embed 
import time

from utils.utils_common import crop_and_merge  
from utils.utils_voxel2mesh.graph_conv import adjacency_matrix, Features2Features, Feature2VertexLayer 
from utils.utils_voxel2mesh.feature_sampling import LearntNeighbourhoodSampling 
from utils.utils_voxel2mesh.file_handle import read_obj 


from utils.utils_unet import UNetLayer
from utils.angle_distortions import angle_distortions
from utils.area_distortions import area_distortions

import wandb
  
def deformation_dist(vertices, faces_prev, N_prev):
    vertices_primary = vertices[0,:N_prev, :]
    vertices_secondary = vertices[0,N_prev:, :]
    faces_primary = faces_prev[0]
    
    edge_combinations_3 = torch.tensor(list(combinations(range(3), 2))).cuda(vertices.device)
    edges = faces_primary[:, edge_combinations_3]
    unique_edges = edges.view(-1, 2)
    unique_edges, _ = torch.sort(unique_edges, dim=1)
    unique_edges, _ = torch.unique(unique_edges, return_inverse=True, dim=0)
    face_edges_primary = vertices_primary[unique_edges]

    a = face_edges_primary[:,0]
    b = face_edges_primary[:,1]
    v = vertices_secondary

    va = v - a
    vb = v - b
    ba = b - a

    cond1 = (va * ba).sum(1)
    norm1 = torch.norm(va, dim=1)

    cond2 = (vb * ba).sum(1)
    norm2 = torch.norm(vb, dim=1)

    dist = torch.norm(torch.cross(va, ba), dim=1)/torch.norm(ba, dim=1)
    dist[cond1 < 0] = norm1[cond1 < 0]
    dist[cond2 < 0] = norm2[cond2 < 0]

    return dist

 
class Voxel2Mesh(nn.Module):
    """ Voxel2Mesh  """
 
    def __init__(self, config):
        super(Voxel2Mesh, self).__init__()

        self.config = config
          
        self.max_pool = nn.MaxPool3d(2) if config.ndims == 3 else nn.MaxPool2d(2) 

        ConvLayer = nn.Conv3d if config.ndims == 3 else nn.Conv2d
        ConvTransposeLayer = nn.ConvTranspose3d if config.ndims == 3 else nn.ConvTranspose2d
        batch_size = config.batch_size
 

        '''  Down layers '''
        down_layers = [UNetLayer(config.num_input_channels, config.first_layer_channels, config.ndims)]
        for i in range(1, config.steps + 1):
            graph_conv_layer = UNetLayer(config.first_layer_channels * 2 ** (i - 1), config.first_layer_channels * 2 ** i, config.ndims)
            down_layers.append(graph_conv_layer)
        self.down_layers = down_layers
        self.encoder = nn.Sequential(*down_layers)
 

        ''' Up layers ''' 
        self.skip_count = []
        self.latent_features_count = []
        for i in range(config.steps+1):
            self.skip_count += [config.first_layer_channels * 2 ** (config.steps-i)] 
            self.latent_features_count += [32]

        dim = 3

        up_std_conv_layers = []
        up_f2f_layers = []
        up_f2v_layers = []
        for i in range(config.steps+1):
            graph_unet_layers = []
            feature2vertex_layers = []
            skip = LearntNeighbourhoodSampling(config, self.skip_count[i], i)
            # lyr = Feature2VertexLayer(self.skip_count[i])
            if i == 0:
                grid_upconv_layer = None
                grid_unet_layer = None
                for k in range(config.num_classes-1):
                    graph_unet_layers += [Features2Features(self.skip_count[i] + dim, self.latent_features_count[i], hidden_layer_count=config.graph_conv_layer_count)] # , graph_conv=GraphConv

            else:
                grid_upconv_layer = ConvTransposeLayer(in_channels=config.first_layer_channels   * 2**(config.steps - i+1), out_channels=config.first_layer_channels * 2**(config.steps-i), kernel_size=2, stride=2)
                grid_unet_layer = UNetLayer(config.first_layer_channels * 2**(config.steps - i + 1), config.first_layer_channels * 2**(config.steps-i), config.ndims, config.batch_norm)
                for k in range(config.num_classes-1):
                    graph_unet_layers += [Features2Features(self.skip_count[i] + self.latent_features_count[i-1] + dim, self.latent_features_count[i], hidden_layer_count=config.graph_conv_layer_count)] #, graph_conv=GraphConv if i < config.steps else GraphConvNoNeighbours

            for k in range(config.num_classes-1):
                feature2vertex_layers += [Feature2VertexLayer(self.latent_features_count[i], 3)] 
 

            up_std_conv_layers.append((skip, grid_upconv_layer, grid_unet_layer))
            up_f2f_layers.append(graph_unet_layers)
            up_f2v_layers.append(feature2vertex_layers)
        
 

        self.up_std_conv_layers = up_std_conv_layers
        self.up_f2f_layers = up_f2f_layers
        self.up_f2v_layers = up_f2v_layers

        self.decoder_std_conv = nn.Sequential(*chain(*up_std_conv_layers))
        self.decoder_f2f = nn.Sequential(*chain(*up_f2f_layers))
        self.decoder_f2v = nn.Sequential(*chain(*up_f2v_layers))

        self.fc1 = nn.Linear(1000*32*3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

        ''' Final layer (for voxel decoder)'''
        self.final_layer = ConvLayer(in_channels=config.first_layer_channels, out_channels=config.num_classes, kernel_size=1)

        sphere_path='./spheres/icosahedron_{}.obj'.format(42)
        sphere_vertices, sphere_faces = read_obj(sphere_path)
        sphere_vertices = torch.from_numpy(sphere_vertices).cuda(self.config.device).float()
        self.sphere_vertices = sphere_vertices/torch.sqrt(torch.sum(sphere_vertices**2,dim=1)[:,None])[None]
        self.sphere_faces = torch.from_numpy(sphere_faces).cuda(self.config.device).long()[None]

 
  
    def forward(self, data):
         
        x = data['x'] 
        unpool_indices = data['unpool'] 

        sphere_vertices = self.sphere_vertices.clone()
        vertices = sphere_vertices.clone()
        faces = self.sphere_faces.clone() 
        batch_size = self.config.batch_size  
 
        # first layer
        x = self.down_layers[0](x)
        down_outputs = [x]

        # down layers
        for unet_layer in self.down_layers[1:]:
            x = self.max_pool(x)
            x = unet_layer(x) 
            down_outputs.append(x)

  
        A, D = adjacency_matrix(vertices, faces)
        pred = [None] * self.config.num_classes 
        for k in range(self.config.num_classes-1):
            pred[k] = [[vertices.clone(), faces.clone(), None, None, sphere_vertices.clone()]]

 
        for i, ((skip_connection, grid_upconv_layer, grid_unet_layer), up_f2f_layers, up_f2v_layers, down_output, skip_amount, do_unpool) in enumerate(zip(self.up_std_conv_layers, self.up_f2f_layers, self.up_f2v_layers, down_outputs[::-1], self.skip_count, unpool_indices)):
            if grid_upconv_layer is not None and i > 0:
                x = grid_upconv_layer(x)
                x = crop_and_merge(down_output, x)
                x = grid_unet_layer(x)
            elif grid_upconv_layer is None:
                x = down_output

            voxel_pred = self.final_layer(x) if i == len(self.up_std_conv_layers)-1 else None

            for k in range(self.config.num_classes-1)[::-1]: # reverse
                nodule_idx = self.config.num_classes-2
                graph_unet_layer = up_f2f_layers[k]
                feature2vertex = up_f2v_layers[k]
                if k == nodule_idx:
                    # load mesh information from previous iteration for class k
                    vertices = pred[k][i][0]
                    faces = pred[k][i][1]
                    latent_features = pred[k][i][2]
                    sphere_vertices = pred[k][i][4]
                else:
                    # load mesh information from current iteration for class nodule
                    vertices = pred[nodule_idx][i+1][0].clone()
                    faces = pred[nodule_idx][i+1][1].clone()
                    if i == 0:
                        latent_features = None
                    else:
                        latent_features = pred[nodule_idx][i+1][2].clone()
                    sphere_vertices = pred[nodule_idx][i+1][4].clone()
 
                if do_unpool[0] == 1 and k == nodule_idx:
                    # Get candidate vertices using uniform unpool
                    mesh = Meshes(verts=list(vertices), faces=list(faces))
                    
                    vert_feats = torch.cat((sphere_vertices, latent_features), 2)[0]

                    subdivide = SubdivideMeshes()
                    mesh, vert_feats = subdivide(mesh, feats=vert_feats)
             
                    vertices = mesh.verts_list()[0][None]
                    faces = mesh.faces_list()[0][None]
                    sphere_vertices = vert_feats[:,:3][None]
                    latent_features = vert_feats[:,3:][None]

                
                A, D = adjacency_matrix(vertices, faces)
                skipped_features = skip_connection(x[:, :skip_amount], vertices)      
                      
                latent_features = torch.cat([latent_features, skipped_features, vertices], dim=2) if latent_features is not None else torch.cat([skipped_features, vertices], dim=2)
 
                latent_features = graph_unet_layer(latent_features, A, D, vertices, faces)
                deltaV = feature2vertex(latent_features, A, D, vertices, faces)
                vertices = vertices + deltaV 

                pred[k] += [[vertices, faces, latent_features, voxel_pred, sphere_vertices]]

            # keep the same mesh after selection
            if do_unpool[0] == 1:
                # load mesh information from previous iteration for class 0 peak
                faces_prev = pred[0][i][1]
                _, N_prev, _ = pred[0][i][0].shape 

                dist = [None] * (self.config.num_classes-1) 
                # Discard the vertices that were introduced from the uniform unpool and didn't deform much
                for k in range(self.config.num_classes-1):
                    # load mesh information from current iteration for all classes
                    vertices = pred[k][i+1][0]
                    dist[k] = deformation_dist(vertices, faces_prev, N_prev)

                dist = torch.max(torch.vstack(dist), 0).values
                sorted_, _ = torch.sort(dist)
                threshold = sorted_[int(0.3*len(sorted_))] 
                selected = torch.cat([torch.arange(N_prev).cuda(self.config.device), (dist > threshold).nonzero()[:,0]+N_prev])

                sphere_vertices = sphere_vertices[0, selected]
                sphere_vertices = sphere_vertices/torch.sqrt(torch.sum(sphere_vertices**2,dim=1)[:,None])
                hull = ConvexHull(sphere_vertices.data.cpu().numpy())  
                faces = torch.from_numpy(hull.simplices).long().cuda(self.config.device)
                sphere_vertices = sphere_vertices[None]
                faces = faces[None]

                for k in range(self.config.num_classes-1):
                    vertices = pred[k][i+1][0][:,selected]
                    latent_features = pred[k][i+1][2][:,selected]
                    pred[k][i+1][0] = vertices
                    pred[k][i+1][1] = faces
                    pred[k][i+1][2] = latent_features
                    pred[k][i+1][4] = sphere_vertices
            
        features = []
        for k in range(self.config.num_classes-1):
            vertices = pred[k][i+1][0]
            faces = pred[k][i+1][1]
            latent_features = pred[k][i+1][2]
            features.append(latent_features[:, 0:1000])
        x = torch.concat(features,2).flatten()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
 
        return pred, x


    def loss(self, data, epoch):     
        pred, output = self.forward(data)  
        # embed()
        
        bce_loss = torch.tensor(0).float().cuda(self.config.device)
        ce_loss = torch.tensor(0).float().cuda(self.config.device)
        chamfer_loss = torch.tensor(0).float().cuda(self.config.device)
        edge_loss = torch.tensor(0).float().cuda(self.config.device)
        laplacian_loss = torch.tensor(0).float().cuda(self.config.device)
        normal_consistency_loss = torch.tensor(0).float().cuda(self.config.device)  
        angle_distortion_loss = torch.tensor(0).float().cuda(self.config.device)  
        area_distortion_loss = torch.tensor(0).float().cuda(self.config.device)  

        target = F.one_hot((data['metadata']['Malignancy']).long().cuda(self.config.device), 2)
        loss = nn.BCELoss()#eight=torch.tensor([0.1, 1]).cuda(self.config.device)  )
        bce_loss += loss(output, target[0].float())

        for c in range(self.config.num_classes-1):
            CE_Loss = nn.CrossEntropyLoss() 
            ce_loss += CE_Loss(pred[c][-1][3], data['y_voxels'])

            target = data['surface_points'][c].cuda(self.config.device) 
            #print(target.size())
            if target.size()[1] == 0:
                continue
            for k, (vertices, faces, _, _, sphere_vertices) in enumerate(pred[c][1:]):
                pred_mesh = Meshes(verts=list(vertices), faces=list(faces))
                sphere_mesh = Meshes(verts=list(sphere_vertices), faces=list(faces))
                angle_d = angle_distortions(pred_mesh.detach(), sphere_mesh.detach())
                area_d = area_distortions(pred_mesh.detach(), sphere_mesh.detach())
                angle_distortion_loss += (angle_d**2).mean()
                area_distortion_loss += (1/(area_d**2+1)).mean()

                pred_points = sample_points_from_meshes(pred_mesh, 1000)
                
                chamfer_loss +=  chamfer_distance(pred_points, target)[0]
                if c == self.config.num_classes-2: #base nodule
                    laplacian_loss +=  mesh_laplacian_smoothing(pred_mesh, method="uniform")
                    normal_consistency_loss += mesh_normal_consistency(pred_mesh) 
                    edge_loss += mesh_edge_loss(pred_mesh) 

        
        
 
        loss = 1 * bce_loss + 1 * chamfer_loss + 1 * ce_loss \
            + 0.1 * laplacian_loss + 1 * edge_loss + 0.1 * normal_consistency_loss \
            #+ 0.001 * angle_distortion_loss #+ 0.01 * area_distortion_loss

 
        log = {
            "loss": loss.detach(),
            "bce_loss": bce_loss.detach(),
            "chamfer_loss": chamfer_loss.detach(), 
            "ce_loss": ce_loss.detach(),
            "normal_consistency_loss": normal_consistency_loss.detach(),
            "edge_loss": edge_loss.detach(),
            "laplacian_loss": laplacian_loss.detach(),
            "angle_distortion_loss": angle_distortion_loss.detach(),
            "area_distortion_loss": area_distortion_loss.detach(),
        }
        return loss, log


 

 

