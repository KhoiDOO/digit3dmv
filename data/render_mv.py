import os
import sys
import glob
import math
import json
import argparse
import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# 1. Camera Trajectory & Transformation Math (Z-Up Standard)
# ==============================================================================

def compute_camera_parameters(azimuth_deg, elevation_deg, dist=2.4, fov_deg=60.0, near=0.1, far=10.0):
    """
    Computes camera position, orientation, View (World-to-Camera), and Projection matrices
    under a Right-Handed Cartesian coordinate system where +Z is UP.
    
    Coordinate conventions:
      - Origin: (0, 0, 0)
      - World Right: +X
      - World Extrusion/Forward: +Y
      - World Up: +Z
      - Front View (azimuth=0, elevation=0): camera at (0, -dist, 0) looking at (0, 0, 0)
    """
    phi = math.radians(azimuth_deg)
    theta = math.radians(elevation_deg)
    
    # Spherical to Cartesian coordinates (Z-Up)
    cx = dist * math.cos(theta) * math.sin(phi)
    cy = -dist * math.cos(theta) * math.cos(phi)
    cz = dist * math.sin(theta)
    
    cam_pos = np.array([cx, cy, cz], dtype=np.float32)
    look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    # Forward vector (from camera to origin)
    f = look_at - cam_pos
    f = f / np.linalg.norm(f)
    
    # World Up vector
    u_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    
    # Right vector
    r = np.cross(f, u_world)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-6:
        # Singularity handling for straight top/bottom views
        u_fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        r = np.cross(f, u_fallback)
        r = r / np.linalg.norm(r)
    else:
        r = r / r_norm
        
    # Camera Up vector
    u = np.cross(r, f)
    u = u / np.linalg.norm(u)
    
    # Rotation matrix R (World to Camera orientation: rows are r, u, -f)
    R_w2c = np.stack([r, u, -f], axis=0) # 3x3
    t_w2c = -R_w2c @ cam_pos            # 3x1
    
    # 4x4 World-to-Camera (w2c) Matrix
    T_w2c = np.eye(4, dtype=np.float32)
    T_w2c[:3, :3] = R_w2c
    T_w2c[:3, 3] = t_w2c
    
    # 4x4 Camera-to-World (c2w) Matrix
    T_c2w = np.linalg.inv(T_w2c)
    
    # 4x4 OpenGL Perspective Projection Matrix for nvdiffrast
    fov_rad = math.radians(fov_deg)
    tan_half_fov = math.tan(fov_rad / 2.0)
    
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 1.0 / tan_half_fov
    P[1, 1] = 1.0 / tan_half_fov
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -(2.0 * far * near) / (far - near)
    P[3, 2] = -1.0
    
    # Model-View-Projection (MVP) Matrix
    MVP = P @ T_w2c
    
    return {
        "azimuth_deg": float(azimuth_deg),
        "elevation_deg": float(elevation_deg),
        "distance": float(dist),
        "camera_position": cam_pos.tolist(),
        "look_at": look_at.tolist(),
        "up_vector": u.tolist(),
        "forward_vector": f.tolist(),
        "right_vector": r.tolist(),
        "rotation_matrix": R_w2c.tolist(),
        "world2cam_matrix": T_w2c.tolist(),
        "cam2world_matrix": T_c2w.tolist(),
        "projection_matrix": P.tolist(),
        "mvp_matrix": MVP.tolist(),
        "cam_pos_np": cam_pos,
        "forward_np": f,
        "mvp_np": MVP,
        "w2c_np": T_w2c,
        "c2w_np": T_c2w
    }

def build_view_trajectories(dist=2.4, fov_deg=60.0, front_delta=30.0):
    """
    Builds the 21 standard camera views:
      - 9 Front views: (0, 0) straight view + 8 surrounding ±front_delta azimuth/elevation shifts
      - 12 360 views: 4 diagonal quadrants at azimuths (45°, 135°, 225°, 315°) across elevations (0°, +45°, -45°)
    """
    views_metadata = {}
    view_list = []
    
    # 1. Front Views (9 total)
    front_angles = [
        (0.0, 0.0),                     # view_00: Straight front
        (-front_delta, -front_delta),   # view_01: Bottom-Left
        (0.0, -front_delta),           # view_02: Bottom-Center
        (front_delta, -front_delta),    # view_03: Bottom-Right
        (-front_delta, 0.0),           # view_04: Center-Left
        (front_delta, 0.0),            # view_05: Center-Right
        (-front_delta, front_delta),    # view_06: Top-Left
        (0.0, front_delta),            # view_07: Top-Center
        (front_delta, front_delta),     # view_08: Top-Right
    ]
    
    for idx, (az, el) in enumerate(front_angles):
        view_key = f"front/view_{idx:02d}"
        params = compute_camera_parameters(az, el, dist=dist, fov_deg=fov_deg)
        params["view_set"] = "front"
        params["view_idx"] = idx
        params["rel_filename"] = f"front/view_{idx:02d}.jpg"
        views_metadata[view_key] = params
        view_list.append((view_key, params))
        
    # 2. 360 Views (12 total)
    c360_angles = [
        # Elevation = 0.0 (Equatorial diagonal views)
        (45.0, 0.0),      # view_00: Front-Right
        (135.0, 0.0),     # view_01: Back-Right
        (225.0, 0.0),     # view_02: Back-Left
        (315.0, 0.0),     # view_03: Front-Left
        # Elevation = +45.0 (Top-down diagonal views)
        (45.0, 45.0),     # view_04: Top-Front-Right
        (135.0, 45.0),    # view_05: Top-Back-Right
        (225.0, 45.0),    # view_06: Top-Back-Left
        (315.0, 45.0),    # view_07: Top-Front-Left
        # Elevation = -45.0 (Bottom-up diagonal views)
        (45.0, -45.0),    # view_08: Bottom-Front-Right
        (135.0, -45.0),   # view_09: Bottom-Back-Right
        (225.0, -45.0),   # view_10: Bottom-Back-Left
        (315.0, -45.0),   # view_11: Bottom-Front-Left
    ]
    
    for idx, (az, el) in enumerate(c360_angles):
        view_key = f"360/view_{idx:02d}"
        params = compute_camera_parameters(az, el, dist=dist, fov_deg=fov_deg)
        params["view_set"] = "360"
        params["view_idx"] = idx
        params["rel_filename"] = f"360/view_{idx:02d}.jpg"
        views_metadata[view_key] = params
        view_list.append((view_key, params))
        
    return views_metadata, view_list


# ==============================================================================
# 2. Fast Wavefront OBJ Loader & Geometry Utilities
# ==============================================================================

def load_obj(obj_path, device="cuda"):
    """
    Fast ASCII OBJ parser extracting vertices and triangle face indices.
    """
    verts, faces = [], []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()
                faces.append([int(parts[1].split('/')[0]) - 1,
                              int(parts[2].split('/')[0]) - 1,
                              int(parts[3].split('/')[0]) - 1])
                
    verts_t = torch.tensor(verts, dtype=torch.float32, device=device)
    faces_t = torch.tensor(faces, dtype=torch.int32, device=device)
    return verts_t, faces_t

def compute_vertex_normals(verts, faces):
    """
    Computes area-weighted vertex normals on CUDA.
    """
    e1 = verts[faces[:, 1]] - verts[faces[:, 0]]
    e2 = verts[faces[:, 2]] - verts[faces[:, 0]]
    fn = torch.linalg.cross(e1, e2)
    
    vn = torch.zeros_like(verts)
    vn.index_add_(0, faces[:, 0], fn)
    vn.index_add_(0, faces[:, 1], fn)
    vn.index_add_(0, faces[:, 2], fn)
    return torch.nn.functional.normalize(vn, dim=-1, eps=1e-8)


# ==============================================================================
# 3. High-Performance Multi-View nvdiffrast Renderer
# ==============================================================================

class MultiViewRenderer:
    def __init__(self, resolution=128, dist=2.4, fov_deg=60.0, front_delta=30.0, box_bound=1.2, device="cuda"):
        self.resolution = resolution
        self.dist = dist
        self.fov_deg = fov_deg
        self.front_delta = front_delta
        self.box_bound = box_bound
        self.device = device
        
        # Initialize headless CUDA rasterizer context
        self.glctx = dr.RasterizeCudaContext(device=device)
        
        # Precompute view parameters
        self.views_metadata, self.view_list = build_view_trajectories(dist=dist, fov_deg=fov_deg, front_delta=front_delta)
        self.num_views = len(self.view_list) # 21 views
        
        # Stack MVP matrices and camera vectors for batched rasterization
        mvp_list = [v[1]["mvp_np"] for v in self.view_list]
        self.mvp_batch = torch.tensor(np.stack(mvp_list, axis=0), dtype=torch.float32, device=device) # (17, 4, 4)
        
        cam_pos_list = [v[1]["cam_pos_np"] for v in self.view_list]
        self.cam_pos_batch = torch.tensor(np.stack(cam_pos_list, axis=0), dtype=torch.float32, device=device) # (17, 3)
        
        forward_list = [v[1]["forward_np"] for v in self.view_list]
        self.forward_batch = torch.tensor(np.stack(forward_list, axis=0), dtype=torch.float32, device=device) # (17, 3)

    def render_mesh(self, verts, faces):
        """
        Renders Normal Maps and Depth Maps for all 17 camera views in a single batched GPU dispatch.
        
        Returns:
          normal_maps_uint8: (17, H, W, 3) numpy array uint8 [0, 255]
          depth_maps_uint8:  (17, H, W)    numpy array uint8 [0, 255]
        """
        N = verts.shape[0]
        V = self.num_views
        
        # 1. Compute vertex world normals
        vn = compute_vertex_normals(verts, faces) # (N, 3)
        
        # 2. Homogeneous coordinates & batched clip transformation
        verts_homo = torch.cat([verts, torch.ones((N, 1), device=self.device)], dim=-1) # (N, 4)
        
        # verts_homo: (1, N, 4), mvp_batch.transpose: (17, 4, 4) -> v_clip: (17, N, 4)
        v_clip = torch.matmul(verts_homo.unsqueeze(0), self.mvp_batch.transpose(1, 2)).contiguous()
        faces = faces.contiguous()
        
        # 3. Single batched rasterization call for all 17 views
        rast, _ = dr.rasterize(self.glctx, v_clip, faces, resolution=[self.resolution, self.resolution])
        rast = rast.contiguous()
        mask = (rast[..., 3:4] > 0) # (17, H, W, 1)
        
        # 4. Interpolate World Normals
        # Broadcast vertex normals across the 17 views
        vn_batch = vn.unsqueeze(0).expand(V, -1, -1).contiguous() # (17, N, 3)
        n_interp, _ = dr.interpolate(vn_batch, rast, faces) # (17, H, W, 3)
        n_interp = torch.nn.functional.normalize(n_interp, dim=-1, eps=1e-8)
        
        # World normal color encoding: (n + 1.0) / 2.0 * 255
        norm_rgb = (n_interp + 1.0) * 0.5
        norm_rgb = torch.where(mask, norm_rgb, torch.zeros_like(norm_rgb))
        # Flip vertically from OpenGL bottom-up NDC coordinates to standard image coordinates
        normal_uint8 = (norm_rgb * 255.0).clamp(0, 255).byte().cpu().numpy()[:, ::-1, :, :] # (17, H, W, 3)
        
        # 5. Compute Linear Depth within Box [-box_bound, box_bound]^3
        # Optical forward distance: z_cam = (v - cam_pos) . forward
        # verts: (1, N, 3), cam_pos_batch: (17, 1, 3) -> diff: (17, N, 3)
        diff = verts.unsqueeze(0) - self.cam_pos_batch.unsqueeze(1) # (17, N, 3)
        v_depth = torch.sum(diff * self.forward_batch.unsqueeze(1), dim=-1, keepdim=True).contiguous() # (17, N, 1)
        
        d_interp, _ = dr.interpolate(v_depth, rast, faces) # (17, H, W, 1)
        
        # Depth normalization: map [dist - box_bound, dist + box_bound] to [0.0, 1.0]
        # For dist=2.4 and box_bound=1.2: [1.2, 3.6] -> [0.0, 1.0]
        min_depth = self.dist - self.box_bound
        max_depth = self.dist + self.box_bound
        depth_span = max_depth - min_depth
        
        d_norm = torch.clamp((d_interp - min_depth) / depth_span, 0.0, 1.0)
        d_norm = torch.where(mask, d_norm, torch.zeros_like(d_norm))
        depth_uint8 = (d_norm.squeeze(-1) * 255.0).clamp(0, 255).byte().cpu().numpy()[:, ::-1, :] # (17, H, W)
        
        return normal_uint8, depth_uint8

    def get_serializable_cameras_json(self):
        """
        Returns JSON-serializable camera trajectory specifications.
        """
        clean_views = {}
        for k, v in self.views_metadata.items():
            clean_views[k] = {
                "view_type": v["view_set"],
                "index": v["view_idx"],
                "azimuth_deg": v["azimuth_deg"],
                "elevation_deg": v["elevation_deg"],
                "distance": v["distance"],
                "camera_position": v["camera_position"],
                "look_at": v["look_at"],
                "up_vector": v["up_vector"],
                "rotation_matrix": v["rotation_matrix"],
                "world2cam_matrix": v["world2cam_matrix"],
                "cam2world_matrix": v["cam2world_matrix"],
                "projection_matrix": v["projection_matrix"],
                "mvp_matrix": v["mvp_matrix"],
                "rel_image_path": v["rel_filename"]
            }
            
        fov_rad = math.radians(self.fov_deg)
        focal_length = (self.resolution / 2.0) / math.tan(fov_rad / 2.0)
        
        return {
            "camera_model": "PINHOLE",
            "coordinate_system": "RIGHT_HANDED_Z_UP",
            "resolution": [self.resolution, self.resolution],
            "fov_deg": self.fov_deg,
            "focal_length_px": focal_length,
            "principal_point_px": [self.resolution / 2.0, self.resolution / 2.0],
            "distance": self.dist,
            "front_delta_deg": self.front_delta,
            "box_bound": self.box_bound,
            "num_views": self.num_views,
            "views": clean_views
        }


# ==============================================================================
# 4. Asynchronous Sample Exporter
# ==============================================================================

def save_single_mesh_multiview(dest_dir, normal_uint8, depth_uint8, views_list, camera_json_dict):
    """
    Saves the rendered normal and depth images and cameras.json for a mesh sample.
    """
    normal_front_dir = os.path.join(dest_dir, "normal/front")
    normal_360_dir = os.path.join(dest_dir, "normal/360")
    depth_front_dir = os.path.join(dest_dir, "depth/front")
    depth_360_dir = os.path.join(dest_dir, "depth/360")
    
    os.makedirs(normal_front_dir, exist_ok=True)
    os.makedirs(normal_360_dir, exist_ok=True)
    os.makedirs(depth_front_dir, exist_ok=True)
    os.makedirs(depth_360_dir, exist_ok=True)
    
    # Save camera metadata JSON
    cameras_json_path = os.path.join(dest_dir, "cameras.json")
    with open(cameras_json_path, 'w') as f:
        json.dump(camera_json_dict, f, indent=2)
        
    # Save all 17 views
    for idx, (view_key, view_params) in enumerate(views_list):
        view_set = view_params["view_set"]
        view_idx = view_params["view_idx"]
        
        filename = f"view_{view_idx:02d}.jpg"
        
        # Normal image
        norm_path = os.path.join(dest_dir, "normal", view_set, filename)
        Image.fromarray(normal_uint8[idx]).save(norm_path, quality=95)
        
        # Depth image
        depth_path = os.path.join(dest_dir, "depth", view_set, filename)
        Image.fromarray(depth_uint8[idx]).save(depth_path, quality=95)


# ==============================================================================
# 5. Dataset Processing Pipeline
# ==============================================================================

def process_dataset(src_dir, dest_dir, renderer, args):
    """
    Processes all OBJ meshes in src_dir and saves multi-view renders to dest_dir.
    """
    os.makedirs(dest_dir, exist_ok=True)
    obj_files = sorted(glob.glob(os.path.join(src_dir, "*.obj")))
    
    if args.debug:
        obj_files = obj_files[:10]
        
    camera_json_dict = renderer.get_serializable_cameras_json()
    views_list = renderer.view_list
    
    print(f"Rendering {len(obj_files)} meshes from {os.path.basename(src_dir)} -> {dest_dir} (Resolution: {args.resolution}x{args.resolution})...")
    
    # Thread pool for asynchronous disk I/O
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        
        for obj_path in tqdm(obj_files, desc=f"Rendering {os.path.basename(src_dir)}"):
            mesh_name = os.path.splitext(os.path.basename(obj_path))[0]
            mesh_dest_dir = os.path.join(dest_dir, mesh_name)
            
            # Skip if already rendered and complete
            if not args.overwrite and os.path.exists(os.path.join(mesh_dest_dir, "cameras.json")):
                continue
                
            try:
                verts, faces = load_obj(obj_path, device="cuda")
                if len(verts) == 0 or len(faces) == 0:
                    continue
                    
                normal_uint8, depth_uint8 = renderer.render_mesh(verts, faces)
                
                # Submit disk write to worker pool
                fut = executor.submit(
                    save_single_mesh_multiview,
                    mesh_dest_dir,
                    normal_uint8,
                    depth_uint8,
                    views_list,
                    camera_json_dict
                )
                futures.append(fut)
                
                # Limit memory pressure by draining futures in chunks
                if len(futures) > 200:
                    for f in futures:
                        f.result()
                    futures = []
                    
            except Exception as e:
                print(f"Error processing {obj_path}: {e}")
                
        # Drain remaining write tasks
        for f in futures:
            f.result()


def main():
    parser = argparse.ArgumentParser(description="Render Multi-View Normal & Depth Maps using nvdiffrast")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"], help="Dataset split to render")
    parser.add_argument("--resolution", type=int, default=128, choices=[64, 128, 256, 512], help="Output image resolution")
    parser.add_argument("--distance", type=float, default=2.4, help="Camera distance from origin")
    parser.add_argument("--fov", type=float, default=60.0, help="Camera Field of View in degrees")
    parser.add_argument("--front_delta", type=float, default=30.0, help="Angular perturbation step in degrees for front camera views (default: 30.0)")
    parser.add_argument("--output_folder", type=str, default=None, help="Custom output directory name or path (e.g. mv_64, mv_128)")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8), help="Number of concurrent I/O disk workers")
    parser.add_argument("--debug", action="store_true", help="Render only first 10 meshes for quick verification")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rendered files")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_train = os.path.join(base_dir, "src/train")
    src_test = os.path.join(base_dir, "src/test")
    
    if args.output_folder is not None:
        if os.path.isabs(args.output_folder):
            mv_root = args.output_folder
        else:
            mv_root = os.path.join(base_dir, args.output_folder)
    elif args.debug:
        mv_root = os.path.join(base_dir, "debug_mv")
    else:
        mv_root = os.path.join(base_dir, f"mv_{args.resolution}")
        
    os.makedirs(mv_root, exist_ok=True)

    print("=" * 60)
    print("  Digit3D Multi-View Normal & Depth Rendering Pipeline")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Camera Distance: {args.distance}, FOV: {args.fov}°, Front Delta: {args.front_delta}°")
    print(f"Output Directory: {mv_root}")
    print(f"Mode: {'DEBUG (10 samples)' if args.debug else 'FULL DATASET'}")
    print("=" * 60)

    # Initialize renderer
    renderer = MultiViewRenderer(resolution=args.resolution, dist=args.distance, fov_deg=args.fov, front_delta=args.front_delta, box_bound=1.2, device="cuda")
    
    # Save canonical global cameras.json
    global_camera_path = os.path.join(mv_root, "cameras.json")
    with open(global_camera_path, 'w') as f:
        json.dump(renderer.get_serializable_cameras_json(), f, indent=2)
    print(f"Saved canonical global camera metadata to {global_camera_path}")

    # Process Splits
    if args.split in ["test", "all"]:
        if os.path.exists(src_test):
            process_dataset(src_test, os.path.join(mv_root, "test"), renderer, args)
        else:
            print(f"Warning: {src_test} does not exist!")
            
    if args.split in ["train", "all"]:
        if os.path.exists(src_train):
            process_dataset(src_train, os.path.join(mv_root, "train"), renderer, args)
        else:
            print(f"Warning: {src_train} does not exist!")

    print("\nMulti-View Rendering Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
