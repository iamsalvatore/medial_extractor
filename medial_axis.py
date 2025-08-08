import numpy as np
import trimesh
import argparse
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from scipy.ndimage import binary_closing, generate_binary_structure, binary_erosion
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import json

class MedialExtractor:
    """
    Medial Axis Extractor.
    """
    
    def __init__(self, mesh, voxel_resolution=256):
        self.mesh = mesh
        self.resolution = voxel_resolution
        
        # Compute mesh properties
        self.bounds_min = mesh.bounds[0]
        self.bounds_max = mesh.bounds[1]
        self.mesh_extents = mesh.extents
        self.mesh_scale = np.linalg.norm(self.mesh_extents)
        
        # Compute voxel size
        max_extent = np.max(self.mesh_extents)
        self.voxel_size = max_extent / voxel_resolution
        self.grid_dims = np.ceil(self.mesh_extents / self.voxel_size).astype(int)
        
        print(f"Simple Continuous Skeleton Extractor")
        print(f"  Resolution: {voxel_resolution}")
        print(f"  Voxel size: {self.voxel_size:.3f}")
        print(f"  Grid: {self.grid_dims}")
        
    def voxelize_mesh(self):
        """Standard voxelization with good coverage."""
        print("\nVoxelizing mesh...")
        
        voxel_grid = np.zeros(self.grid_dims, dtype=bool)
        
        # Voxelize vertices
        vertex_voxels = np.floor((self.mesh.vertices - self.bounds_min) / self.voxel_size).astype(int)
        vertex_voxels = np.clip(vertex_voxels, 0, self.grid_dims - 1)
        voxel_grid[vertex_voxels[:, 0], vertex_voxels[:, 1], vertex_voxels[:, 2]] = True
        
        # Voxelize triangles for better coverage
        print("  Voxelizing triangles...")
        for face_idx in tqdm(range(0, len(self.mesh.faces), 1000), desc="Face batches"):
            faces_batch = self.mesh.faces[face_idx:face_idx+1000]
            
            for face in faces_batch:
                v0, v1, v2 = self.mesh.vertices[face]
                
                # Sample triangle
                for u in np.linspace(0, 1, 4):
                    for v in np.linspace(0, 1 - u, 4):
                        if u + v <= 1:
                            w = 1 - u - v
                            point = u * v0 + v * v1 + w * v2
                            voxel = np.floor((point - self.bounds_min) / self.voxel_size).astype(int)
                            voxel = np.clip(voxel, 0, self.grid_dims - 1)
                            voxel_grid[voxel[0], voxel[1], voxel[2]] = True
        
        return voxel_grid
    
    def fill_interior(self, surface_voxels):
        """Fill interior with resolution-adaptive parameters."""
        print("  Filling interior...")
        
        # Adaptive parameters based on resolution
        # Higher resolution needs more aggressive closing
        if self.resolution >= 512:
            closing_iterations = 3
            erosion_size = 2
        elif self.resolution >= 256:
            closing_iterations = 2
            erosion_size = 1
        else:
            closing_iterations = 1
            erosion_size = 0
        
        # Close surface gaps
        structure = generate_binary_structure(3, 1)
        closed = binary_closing(surface_voxels, structure=structure, iterations=closing_iterations)
        
        # Fill along all axes
        filled = closed.copy()
        
        # Multiple passes for robustness
        for _ in range(2):
            # Z-axis
            for z in range(filled.shape[2]):
                if np.any(filled[:, :, z]):
                    filled[:, :, z] = binary_fill_holes(filled[:, :, z])
            
            # Y-axis
            for y in range(filled.shape[1]):
                if np.any(filled[:, y, :]):
                    filled[:, y, :] = binary_fill_holes(filled[:, y, :])
            
            # X-axis
            for x in range(filled.shape[0]):
                if np.any(filled[x, :, :]):
                    filled[x, :, :] = binary_fill_holes(filled[x, :, :])
        
        # Erode to remove near-surface regions at high resolution
        if erosion_size > 0:
            print(f"    Applying erosion (size={erosion_size}) to remove surface artifacts...")
            structure = generate_binary_structure(3, erosion_size)
            filled = binary_erosion(filled, structure=structure, iterations=1)
        
        return filled
    
    def extract_skeleton(self):
        """Extract skeleton with connectivity information."""
        # Step 1: Voxelize
        surface_voxels = self.voxelize_mesh()
        
        # Step 2: Fill interior
        filled_voxels = self.fill_interior(surface_voxels)
        
        # Step 3: Direct skeletonization
        print("\nExtracting skeleton...")
        print("  Direct skeletonization of filled volume...")
        skeleton_voxels = skeletonize(filled_voxels)
        
        # Step 4: Extract connectivity from voxel skeleton
        print("  Extracting connectivity...")
        skeleton_graph, voxel_to_index = self.extract_voxel_connectivity(skeleton_voxels)
        
        # Convert voxel coordinates to world coordinates
        skeleton_points = np.argwhere(skeleton_voxels)
        skeleton_coords = skeleton_points * self.voxel_size + self.bounds_min
        
        print(f"  Initial skeleton: {len(skeleton_coords)} points")
        print(f"  Graph edges: {skeleton_graph.number_of_edges()}")
        
        # Step 5: Ensure connectivity and clean up
        if len(skeleton_coords) > 0:
            skeleton_coords, skeleton_graph = self.ensure_connectivity_graph(skeleton_coords, skeleton_graph)
        
        # Store graph for later use
        self.skeleton_graph = skeleton_graph
        self.skeleton_coords = skeleton_coords
        
        # Step 6: Extract paths
        self.extract_skeleton_paths()
        
        print(f"\nFinal skeleton: {len(skeleton_coords)} points, {skeleton_graph.number_of_edges()} edges")
        
        return skeleton_coords
    
    
    def extract_voxel_connectivity(self, skeleton_voxels):
        """Extract connectivity graph from voxel skeleton."""
        # Get all skeleton voxel coordinates
        skeleton_points = np.argwhere(skeleton_voxels)
        
        # Create mapping from voxel coordinates to index
        voxel_to_index = {}
        for idx, point in enumerate(skeleton_points):
            voxel_to_index[tuple(point)] = idx
        
        # Build graph
        G = nx.Graph()
        for idx in range(len(skeleton_points)):
            G.add_node(idx)
        
        # Check 26-connectivity for each skeleton voxel
        for idx, point in enumerate(skeleton_points):
            x, y, z = point
            
            # Check all 26 neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        
                        neighbor = (x + dx, y + dy, z + dz)
                        
                        # Check if neighbor is in skeleton
                        if neighbor in voxel_to_index:
                            neighbor_idx = voxel_to_index[neighbor]
                            if neighbor_idx > idx:  # Avoid duplicate edges
                                G.add_edge(idx, neighbor_idx)
        
        return G, voxel_to_index
    
    def ensure_connectivity_graph(self, skeleton_coords, skeleton_graph):
        """Ensure skeleton graph is connected by adding bridge edges."""
        print("\nEnsuring connectivity...")
        
        # Find connected components
        components = list(nx.connected_components(skeleton_graph))
        print(f"  Found {len(components)} components")
        
        if len(components) == 1:
            print("  Already connected!")
            return skeleton_coords, skeleton_graph
        
        # Get largest component
        components = sorted(components, key=len, reverse=True)
        main_component = components[0]
        
        print(f"  Main component has {len(main_component)} points")
        
        new_points = []
        new_edges = []
        
        # Connect other significant components to main
        for comp_idx, component in enumerate(components[1:], 1):
            if len(component) < 10:  # Skip tiny components
                continue
            
            # Find closest points between components
            main_points = skeleton_coords[list(main_component)]
            comp_points = skeleton_coords[list(component)]
            
            main_tree = cKDTree(main_points)
            comp_tree = cKDTree(comp_points)
            
            min_dist = float('inf')
            best_pair = None
            best_indices = None
            
            # Sample points for efficiency
            sample_size = min(100, len(main_component))
            sample_indices = np.random.choice(list(main_component), sample_size, replace=False)
            
            for main_idx in sample_indices:
                p1 = skeleton_coords[main_idx]
                dist, comp_local_idx = comp_tree.query(p1)
                if dist < min_dist:
                    min_dist = dist
                    comp_idx_global = list(component)[comp_local_idx]
                    best_pair = (p1, skeleton_coords[comp_idx_global])
                    best_indices = (main_idx, comp_idx_global)
            
            # Connect if reasonably close
            if best_pair and min_dist < self.mesh_scale * 0.1:
                p1, p2 = best_pair
                idx1, idx2 = best_indices
                
                # Add intermediate points
                num_points = max(2, int(min_dist / self.voxel_size))
                intermediate_indices = []
                
                for i in range(1, num_points):
                    t = i / num_points
                    new_point = (1 - t) * p1 + t * p2
                    new_points.append(new_point)
                    new_idx = len(skeleton_coords) + len(new_points) - 1
                    intermediate_indices.append(new_idx)
                    skeleton_graph.add_node(new_idx)
                
                # Add edges to form path
                if intermediate_indices:
                    skeleton_graph.add_edge(idx1, intermediate_indices[0])
                    for i in range(len(intermediate_indices) - 1):
                        skeleton_graph.add_edge(intermediate_indices[i], intermediate_indices[i+1])
                    skeleton_graph.add_edge(intermediate_indices[-1], idx2)
                else:
                    skeleton_graph.add_edge(idx1, idx2)
                
                print(f"  Connected component {comp_idx} (distance: {min_dist:.2f})")
        
        # Add new points to coordinates
        if new_points:
            skeleton_coords = np.vstack([skeleton_coords, new_points])
            print(f"  Added {len(new_points)} connecting points")
        
        return skeleton_coords, skeleton_graph
    
    def extract_skeleton_paths(self):
        """Extract meaningful paths from the skeleton graph."""
        if not hasattr(self, 'skeleton_graph') or self.skeleton_graph.number_of_nodes() == 0:
            print("  No skeleton graph to process")
            return
        
        # Identify endpoints and junctions
        self.endpoints = []
        self.junctions = []
        
        for node in self.skeleton_graph.nodes():
            degree = self.skeleton_graph.degree(node)
            if degree == 1:
                self.endpoints.append(node)
            elif degree > 2:
                self.junctions.append(node)
        
        print(f"  Found {len(self.endpoints)} endpoints and {len(self.junctions)} junctions")
        
        # Extract the longest path (main centerline)
        self.main_path = self.find_longest_path()
        if self.main_path:
            print(f"  Main path length: {len(self.main_path)} points")
        
        # Extract all branch paths
        self.branch_paths = self.extract_branch_paths()
        print(f"  Found {len(self.branch_paths)} branch paths")
    
    def find_longest_path(self):
        """Find the longest path in the skeleton (likely the main centerline)."""
        if not self.endpoints:
            # If no endpoints, find longest path between any two nodes
            if self.skeleton_graph.number_of_nodes() < 2:
                return list(self.skeleton_graph.nodes())
            
            # Sample nodes to find approximate longest path
            nodes = list(self.skeleton_graph.nodes())
            sample_size = min(20, len(nodes))
            sample_nodes = np.random.choice(nodes, sample_size, replace=False)
            
            longest_path = []
            max_length = 0
            
            for start in sample_nodes:
                # BFS to find farthest node from start
                distances = nx.single_source_shortest_path_length(self.skeleton_graph, start)
                farthest = max(distances, key=distances.get)
                
                if distances[farthest] > max_length:
                    try:
                        path = nx.shortest_path(self.skeleton_graph, start, farthest)
                        if len(path) > max_length:
                            longest_path = path
                            max_length = len(path)
                    except nx.NetworkXNoPath:
                        continue
            
            return longest_path
        
        # Find longest path between endpoints
        longest_path = []
        max_length = 0
        
        for i, start in enumerate(self.endpoints):
            for end in self.endpoints[i+1:]:
                try:
                    path = nx.shortest_path(self.skeleton_graph, start, end)
                    if len(path) > max_length:
                        longest_path = path
                        max_length = len(path)
                except nx.NetworkXNoPath:
                    continue
        
        return longest_path
    
    def extract_branch_paths(self):
        """Extract all significant branch paths."""
        branch_paths = []
        
        # For each junction, find paths to endpoints or other junctions
        for junction in self.junctions:
            # Find paths from this junction to endpoints
            for endpoint in self.endpoints:
                try:
                    path = nx.shortest_path(self.skeleton_graph, junction, endpoint)
                    if len(path) > 5:  # Only keep significant branches
                        branch_paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return branch_paths
    
    def get_ordered_centerline(self, start_point=None, end_point=None):
        """Get an ordered centerline path suitable for camera following.
        
        Args:
            start_point: Optional 3D coordinate to start from (finds nearest skeleton point)
            end_point: Optional 3D coordinate to end at (finds nearest skeleton point)
        
        Returns:
            Ordered list of 3D coordinates along the centerline
        """
        if not hasattr(self, 'main_path') or not self.main_path:
            return self.skeleton_coords
        
        path_indices = self.main_path
        
        # If start/end points specified, find nearest skeleton points
        if start_point is not None or end_point is not None:
            tree = cKDTree(self.skeleton_coords)
            
            if start_point is not None:
                _, start_idx = tree.query(start_point)
            else:
                start_idx = path_indices[0]
            
            if end_point is not None:
                _, end_idx = tree.query(end_point)
            else:
                end_idx = path_indices[-1]
            
            # Find path between specified points
            try:
                path_indices = nx.shortest_path(self.skeleton_graph, start_idx, end_idx)
            except nx.NetworkXNoPath:
                print("Warning: No path found between specified points")
                path_indices = self.main_path
        
        # Return ordered coordinates
        return self.skeleton_coords[path_indices]
    
    def ensure_connectivity(self, skeleton_coords):
        """Ensure skeleton is connected by filling gaps."""
        print("\nEnsuring connectivity...")
        
        tree = cKDTree(skeleton_coords)
        
        # Check current connectivity
        import networkx as nx
        G = nx.Graph()
        for i in range(len(skeleton_coords)):
            G.add_node(i)
        
        # Connect points within 2 voxels
        connect_dist = self.voxel_size * 2.0
        for i in range(len(skeleton_coords)):
            neighbors = tree.query_ball_point(skeleton_coords[i], connect_dist)
            for j in neighbors:
                if j > i:
                    G.add_edge(i, j)
        
        # Find components
        components = list(nx.connected_components(G))
        print(f"  Found {len(components)} components")
        
        if len(components) == 1:
            print("  Already connected!")
            return skeleton_coords
        
        # Connect components
        new_points = []
        
        # Get largest component
        components = sorted(components, key=len, reverse=True)
        main_component = components[0]
        main_points = skeleton_coords[list(main_component)]
        
        print(f"  Main component has {len(main_component)} points")
        
        # Connect other significant components to main
        for comp_idx, component in enumerate(components[1:], 1):
            if len(component) < 10:  # Skip tiny components
                continue
                
            comp_points = skeleton_coords[list(component)]
            
            # Find closest points between main and this component
            comp_tree = cKDTree(comp_points)
            
            min_dist = float('inf')
            best_pair = None
            
            # Sample main component points
            sample_size = min(100, len(main_points))
            sample_indices = np.random.choice(len(main_points), sample_size, replace=False)
            
            for idx in sample_indices:
                p1 = main_points[idx]
                dist, nearest_idx = comp_tree.query(p1)
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (p1, comp_points[nearest_idx])
            
            # Connect if reasonably close
            if best_pair and min_dist < self.mesh_scale * 0.1:
                p1, p2 = best_pair
                
                # Add connecting points
                num_points = int(min_dist / self.voxel_size) + 1
                for i in range(1, num_points):
                    t = i / num_points
                    new_point = (1 - t) * p1 + t * p2
                    new_points.append(new_point)
                
                print(f"  Connected component {comp_idx} (distance: {min_dist:.2f})")
        
        # Add new points
        if new_points:
            skeleton_coords = np.vstack([skeleton_coords, new_points])
            print(f"  Added {len(new_points)} connecting points")
        
        return skeleton_coords
    
    def visualize(self, skeleton_coords, show=True, save_path=None):
        """Visualize skeleton with connectivity."""
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        if len(skeleton_coords) > 0:
            # Draw edges first (behind points)
            if hasattr(self, 'skeleton_graph'):
                for edge in self.skeleton_graph.edges():
                    p1 = skeleton_coords[edge[0]]
                    p2 = skeleton_coords[edge[1]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                           'gray', alpha=0.3, linewidth=0.5)
            
            # Highlight main path if available
            if hasattr(self, 'main_path') and self.main_path:
                main_coords = skeleton_coords[self.main_path]
                ax.plot(main_coords[:, 0], main_coords[:, 1], main_coords[:, 2],
                       'g-', linewidth=2, alpha=0.8, label='Main centerline')
            
            # Draw all skeleton points
            ax.scatter(skeleton_coords[:, 0], skeleton_coords[:, 1], skeleton_coords[:, 2], 
                      c='red', s=3, alpha=0.6, label='Skeleton points')
            
            # Highlight endpoints and junctions
            if hasattr(self, 'endpoints') and self.endpoints:
                endpoint_coords = skeleton_coords[self.endpoints]
                ax.scatter(endpoint_coords[:, 0], endpoint_coords[:, 1], endpoint_coords[:, 2],
                          c='blue', s=30, marker='o', label=f'Endpoints ({len(self.endpoints)})')
            
            if hasattr(self, 'junctions') and self.junctions:
                junction_coords = skeleton_coords[self.junctions]
                ax.scatter(junction_coords[:, 0], junction_coords[:, 1], junction_coords[:, 2],
                          c='yellow', s=40, marker='^', label=f'Junctions ({len(self.junctions)})')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        title = f'Connected Skeleton ({len(skeleton_coords)} points'
        if hasattr(self, 'skeleton_graph'):
            title += f', {self.skeleton_graph.number_of_edges()} edges)'
        else:
            title += ')'
        ax.set_title(title)
        
        # Equal aspect ratio
        if len(skeleton_coords) > 0:
            max_range = np.max(skeleton_coords.max(axis=0) - skeleton_coords.min(axis=0)) / 2
            mid = (skeleton_coords.max(axis=0) + skeleton_coords.min(axis=0)) / 2
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        ax.legend(loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_skeleton(self, skeleton_coords, output_path):
        """Save skeleton with connectivity information."""
        if len(skeleton_coords) == 0:
            print("No skeleton points to save!")
            return
        
        # Save PLY with edges if possible
        base_path = output_path.rsplit('.', 1)[0]
        
        # Save as PLY with custom format including edges
        self.save_ply_with_edges(skeleton_coords, output_path)
        
        # Save connectivity as JSON (includes main_path for centerline)
        json_path = base_path + '_graph.json'
        self.save_graph_json(json_path)
        
        print(f"Saved skeleton with {len(skeleton_coords)} points and {self.skeleton_graph.number_of_edges()} edges")
    
    def save_ply_with_edges(self, skeleton_coords, output_path):
        """Save skeleton as PLY file with edge connectivity."""
        num_vertices = len(skeleton_coords)
        num_edges = self.skeleton_graph.number_of_edges()
        
        with open(output_path, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element edge {num_edges}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write("end_header\n")
            
            # Write vertices
            for coord in skeleton_coords:
                f.write(f"{coord[0]} {coord[1]} {coord[2]} 255 0 0\n")
            
            # Write edges
            for edge in self.skeleton_graph.edges():
                f.write(f"{edge[0]} {edge[1]}\n")
        
        print(f"  Saved PLY with edges to {output_path}")
    
    def save_graph_json(self, json_path):
        """Save skeleton graph as JSON for easy loading."""
        # Convert numpy int64 to regular Python int for JSON serialization
        def convert_to_int(lst):
            if isinstance(lst, list):
                return [convert_to_int(item) for item in lst]
            elif isinstance(lst, (np.integer, np.int64)):
                return int(lst)
            else:
                return lst
        
        graph_data = {
            'vertices': self.skeleton_coords.tolist(),
            'edges': [list(edge) for edge in self.skeleton_graph.edges()],
            'endpoints': convert_to_int(self.endpoints) if hasattr(self, 'endpoints') else [],
            'junctions': convert_to_int(self.junctions) if hasattr(self, 'junctions') else [],
            'main_path': convert_to_int(self.main_path) if hasattr(self, 'main_path') else [],
            'branch_paths': convert_to_int(self.branch_paths) if hasattr(self, 'branch_paths') else []
        }
        
        with open(json_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"  Saved graph data to {json_path}")
    


def main():
    parser = argparse.ArgumentParser(description='Simple continuous skeleton extraction')
    parser.add_argument('input_mesh', help='Input mesh file')
    parser.add_argument('-o', '--output', default='skeleton.ply', help='Output file')
    parser.add_argument('-r', '--resolution', type=int, default=256,
                       help='Resolution (128, 256, or 512 recommended)')
    parser.add_argument('--no_viz', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    # Load mesh
    print(f"Loading mesh from {args.input_mesh}...")
    mesh = trimesh.load(args.input_mesh)
    
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    # Extract skeleton
    extractor = MedialExtractor(mesh, voxel_resolution=args.resolution)
    skeleton_coords = extractor.extract_skeleton()
    
    # Save and visualize
    if len(skeleton_coords) > 0:
        extractor.save_skeleton(skeleton_coords, args.output)
        
        if not args.no_viz:
            extractor.visualize(skeleton_coords, show=True)
    

if __name__ == "__main__":
    main()

    