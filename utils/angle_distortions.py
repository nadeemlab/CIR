import torch
from pytorch3d.structures import Meshes 

def angle_distortions(mesh_orig, mesh_param, is_vertices = False):
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
        angle_orig = compute_per_face_min_angle(verts_packed_orig,faces_packed_orig)
        angle_param = compute_per_face_min_angle(verts_packed_param,faces_packed_param)
        log_angle_param = torch.log(angle_orig/angle_param)

        if is_vertices:
            # to vertices
            F = faces_packed_orig
            A = log_angle_param
            m = torch.arange(F.size(0), device = F.device)
            U = torch.sparse_coo_tensor(torch.vstack([m.repeat(3), F.flatten()]), A.repeat(3))
            log_angle_param = torch.sparse.sum(U, 0).to_dense()

    return log_angle_param
    
    # angle distortion histogram

    #figure
    #[n,x]=hist(log_angle_param, 80)
    #bar(x,n./sum(n),.5,'hist')


##########################################
##########################################
##########################################
def compute_per_face_min_angle(V,F):
    u = V[F[:,[1,2,0]],] - V[F[:,[0,1,2]],]
    v = V[F[:,[2,0,1]],] - V[F[:,[0,1,2]],]

    # normailize
    u = u/torch.norm(u,dim=2).reshape([-1,3,1])
    v = v/torch.norm(v,dim=2).reshape([-1,3,1])
    
    # compute angles
    A = torch.acos(torch.sum(u*v, 2))
    A, _ = A.min(1)
    return A

