import torch
from pytorch3d.structures import Meshes 

def area_distortions(mesh_orig, mesh_param):
    if mesh_orig.isempty() or mesh_param.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=mesh_orig.device, requires_grad=True
        )

    N = len(mesh_orig)
    M = len(mesh_param)
    assert N == M, "Different numbers of meshes"
    verts_packed_orig = mesh_orig.verts_packed()  # (sum(V_n), 3)
    faces_packed_orig = mesh_orig.faces_packed()  # (sum(F_n), 3)
    verts_packed_param = mesh_param.verts_packed()  # (sum(V_n), 3)
    faces_packed_param = mesh_param.faces_packed()  # (sum(F_n), 3)

    num_verts_per_mesh_orig = mesh_orig.num_verts_per_mesh()  # (N,)
    #verts_packed_idx_orig = mesh_orig.verts_packed_to_mesh_idx()  # (sum(V_n),)
    num_verts_per_mesh_param = mesh_param.num_verts_per_mesh()  # (N,)
    #verts_packed_idx_param = mesh_param.verts_packed_to_mesh_idx()  # (sum(V_n),)
    assert num_verts_per_mesh_orig == num_verts_per_mesh_param, "Different numbers of vertices"
    #weights = num_verts_per_mesh_orig.gather(0, verts_packed_idx_orig)  # (sum(V_n),)
    #weights = 1.0 / weights.float()

    with torch.no_grad():
        area_orig = compute_per_vertex_area(verts_packed_orig,faces_packed_orig)
        area_param = compute_per_vertex_area(verts_packed_param,faces_packed_param)
        log_area_param = torch.log(area_param/area_orig)

    return log_area_param
    
    # area distortion histogram

    #figure
    #[n,x]=hist(log_area_param, 80)
    #bar(x,n./sum(n),.5,'hist')


##########################################
##########################################
##########################################
def compute_per_vertex_area(V,F):
    ## Compute area around each V

    # area of each face
    a = V[F[:,2],] - V[F[:,0],]
    b = V[F[:,1],] - V[F[:,0],]
    ab = torch.cross(a,b)
    Af = torch.sqrt(torch.sum(ab**2,1))
    # area of each vertex
    m = torch.arange(F.size(0), device = V.device)
    U = torch.sparse_coo_tensor(torch.vstack([m.repeat(3), F.flatten()]), Af.repeat(3))
    Av = torch.sparse.sum(U, 0).to_dense()

    return Av
