# Medial Axis Extractor

A Python tool for extracting the medial axis (skeleton) of 3D meshes with **full connectivity preservation**, enabling direct path following without interpolation gaps.

## Overview

The Medial Axis Extractor computes a connected skeletal graph representation of 3D mesh objects by:
1. Voxelizing the input mesh into a 3D grid
2. Filling the interior volume
3. Applying 3D skeletonization to extract the medial axis
4. **Preserving connectivity as a graph structure**
5. Identifying junctions, endpoints, and the main centerline path

This approach provides a connected skeletal graph that captures the essential topological structure of the input mesh, perfect for applications like camera path following, vessel navigation, or structural analysis.

## Key Features

- **Connected Skeleton Graph**: Maintains full connectivity between skeleton points using NetworkX
- **No Interpolation Gaps**: Direct traversal along skeleton edges without guessing
- **Graph Analysis**: Automatic detection of endpoints, junctions, and main centerline path
- **Camera Path Following**: Built-in support for smooth camera animation along skeleton
- **Multiple Export Formats**: PLY with edges + JSON graph structure
- **Resolution Adaptive**: Automatic parameter adjustment based on voxel resolution
- **Robust Connectivity**: Automatic component connection within distance thresholds

## Algorithm Details

The medial axis extractor uses a robust pipeline to create connected skeletal graphs:

### 1. Voxelization
The algorithm converts the input mesh into a voxel grid with configurable resolution:
- **Vertex sampling**: All mesh vertices are rasterized into the voxel grid
- **Triangle sampling**: Dense sampling of triangle faces using barycentric coordinates
- **Automatic grid sizing**: Grid dimensions computed based on mesh bounds and resolution
- **Quality preservation**: Higher resolution captures fine details

### 2. Interior Filling with Adaptive Parameters
The voxelized surface is processed to create a solid volume:
- **Resolution-adaptive morphological closing**: More aggressive gap filling at higher resolutions
- **Multi-directional flood filling**: Fills interior along X, Y, and Z axes
- **Erosion for artifact removal**: Optional surface erosion to reduce noise at high resolution
- **Multiple passes**: Ensures complete and robust interior filling

### 3. 3D Skeletonization
The filled volume undergoes topological thinning to extract the medial axis:
- **Scikit-image skeletonization**: Uses proven 3D thinning algorithm
- **Topology preservation**: Maintains essential shape structure
- **Connected result**: Produces a thin, connected skeleton

### 4. Connectivity Extraction
The skeleton voxels are converted to a graph structure:
- **26-connectivity analysis**: Checks all 26 neighboring voxel positions
- **NetworkX graph creation**: Builds graph with skeleton points as nodes
- **Edge preservation**: Maintains connectivity between adjacent skeleton voxels
- **World coordinate conversion**: Transforms voxel coordinates to real-world coordinates

### 5. Graph Analysis and Path Extraction
The skeleton graph is analyzed to extract meaningful structures:
- **Degree analysis**:
  - **Endpoints**: Nodes with degree 1 (terminations)
  - **Junctions**: Nodes with degree > 2 (branch points)
- **Main centerline identification**: Finds longest path between endpoints using graph algorithms
- **Branch path extraction**: Identifies all paths from junctions to endpoints
- **Path ordering**: Creates ordered sequences for direct traversal

### 6. Connectivity Enforcement
The algorithm ensures the skeleton forms a single connected component:
- **Component detection**: Uses NetworkX to find disconnected components
- **Distance-based connection**: Connects components within `2 * voxel_size` threshold
- **Gap bridging**: Adds interpolated points to connect close components
- **Graph validation**: Ensures final result is fully connected

## Installation

### Requirements
```bash
pip install numpy scipy trimesh scikit-image matplotlib networkx tqdm
```

### Dependencies
- `numpy`: Numerical operations
- `scipy`: Spatial data structures and image processing
- `trimesh`: 3D mesh loading and manipulation
- `scikit-image`: Skeletonization algorithm
- `matplotlib`: Visualization
- `networkx`: Graph connectivity analysis
- `tqdm`: Progress bars

## Usage

### Command Line Interface

#### Skeleton Extraction
Basic usage:
```bash
python medial_axis.py input_mesh.obj
```

With custom output and resolution:
```bash
python medial_axis.py input_mesh.stl -o skeleton.ply -r 256
```

Without visualization:
```bash
python medial_axis.py input_mesh.obj --no_viz
```

#### Camera Path Following
Generate smooth camera path from skeleton:
```bash
python use_centerline.py skeleton_base_name --smooth_samples 200
```

Export camera animation keyframes:
```bash
python use_centerline.py skeleton_base_name --export_animation camera_path.json --no_viz
```

### Command Line Arguments

#### medial_axis.py
- `input_mesh`: Path to input mesh file (required)
  - Supports common formats: OBJ, STL, PLY, etc.
- `-o, --output`: Output file path (default: `skeleton.ply`)
- `-r, --resolution`: Voxel grid resolution (default: 256)
  - **128**: Fast processing, suitable for simple shapes
  - **256**: Balanced quality and speed (recommended)
  - **512**: High detail, longer processing time
- `--no_viz`: Skip visualization window

#### use_centerline.py
- `skeleton_files`: Base name of skeleton files (without extension)
- `--smooth_samples`: Number of samples for smooth path (default: 200)
- `--export_animation`: Export animation keyframes to JSON file
- `--no_viz`: Skip visualization

### Python API

```python
import trimesh
from medial_axis import MedialExtractor

# Load mesh
mesh = trimesh.load('model.obj')

# Create extractor with custom resolution
extractor = MedialExtractor(mesh, voxel_resolution=256)

# Extract skeleton with connectivity
skeleton_coords = extractor.extract_skeleton()

# Access graph structure
skeleton_graph = extractor.skeleton_graph  # NetworkX graph
endpoints = extractor.endpoints            # List of endpoint indices
junctions = extractor.junctions            # List of junction indices
main_path = extractor.main_path           # Ordered indices of main centerline

# Get ordered centerline for path following
centerline = extractor.get_ordered_centerline()

# Save results with both PLY and JSON export
extractor.save_ply_with_edges(skeleton_coords, 'skeleton.ply')
extractor.save_graph_json('skeleton_graph.json')

# Visualize with connectivity
extractor.visualize_skeleton(show=True, save_path='skeleton.png')
```

## Output Files

For an input mesh processed as `skeleton.ply`, the tool generates:

### 1. skeleton.ply - PLY with Edge Connectivity
```
ply
format ascii 1.0
element vertex N
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element edge M
property int vertex1
property int vertex2
end_header
```
- Contains both vertices and edges
- Can be visualized in tools supporting edge rendering
- Red colored skeleton points

### 2. skeleton_graph.json - Complete Graph Structure
```json
{
  "vertices": [[x, y, z], ...],      // 3D coordinates
  "edges": [[v1, v2], ...],          // Connectivity pairs
  "endpoints": [idx1, idx2, ...],     // Degree-1 vertices
  "junctions": [idx3, idx4, ...],     // Degree-3+ vertices
  "main_path": [ordered indices],     // Main centerline
  "branch_paths": [[path1], ...]      // Branch paths
}
```

## Camera Path Following

The `use_centerline.py` script enables smooth camera animation along the skeleton:

1. **Loads connected skeleton** from JSON graph
2. **Extracts main centerline** using pre-computed path
3. **Creates smooth interpolation** with cubic splines
4. **Computes camera orientations** (look-at directions)
5. **Exports animation keyframes** for external tools

Example animation export format:
```json
{
  "fps": 30,
  "duration": 10,
  "frames": [
    {
      "frame": 0,
      "time": 0.0,
      "position": [x, y, z],
      "forward": [fx, fy, fz]
    },
    ...
  ]
}
```

## Parameters and Tuning

### Resolution
- **128**: Fast processing, suitable for simple shapes
- **256**: Balanced quality and speed (default)
- **512**: High detail, longer processing time

### Connectivity Distance
The algorithm connects skeleton components within `2 * voxel_size` distance.

### Smooth Path Samples
Controls the number of points in the interpolated camera path (default: 200).

## Applications

- **Medical Imaging**: Virtual endoscopy, vessel navigation
- **Animation**: Camera path planning along skeletal structures
- **Robotics**: Path planning through tubular structures
- **Analysis**: Structural topology, shape analysis
- **Games**: Procedural level generation, navigation meshes

## Advantages Over Point Cloud Skeletons

Traditional skeleton extraction produces disconnected point clouds requiring interpolation. This tool provides:

1. **Full Connectivity**: Every point connected to its neighbors
2. **Graph Structure**: Navigate using graph algorithms
3. **No Guessing**: Direct path following without interpolation
4. **Branch Awareness**: Handle junctions systematically
5. **Ordered Paths**: Pre-computed centerline ordering

## Examples

### Airway Navigation
```bash
# Extract skeleton with connectivity
python medial_axis.py airway.obj -o airway_skeleton.ply -r 256

# Generate camera path for virtual bronchoscopy
python use_centerline.py airway_skeleton --export_animation bronchoscopy.json
```

### Vascular System
```bash
# High-resolution skeleton for detailed vessels
python medial_axis.py vessels.stl -r 512 -o vessels_skeleton.ply

# Create smooth navigation path
python use_centerline.py vessels_skeleton --smooth_samples 500
```

## Tips

1. **Start with lower resolution** (128) for testing, then increase for final results
2. **Check mesh quality** - ensure the mesh is manifold and has consistent normals
3. **Use the JSON graph** for custom path planning algorithms
4. **Adjust smooth_samples** based on path length and desired smoothness
5. **Visualize first** to verify skeleton quality before processing

## Troubleshooting

### Empty skeleton
- Check if mesh is properly loaded and has valid geometry
- Ensure mesh is not inside-out (incorrect normals)
- Try increasing resolution

### Disconnected skeleton
- Algorithm automatically connects components within reasonable distance
- Check console output for "Found X components" message
- If many components remain, verify mesh is watertight and manifold
- Most results show "Found 1 components - Already connected!" indicating success

### Jagged camera path
- Increase `smooth_samples` parameter
- Check if main path was correctly identified
- Verify skeleton connectivity in visualization

### Slow processing
- Reduce resolution
- Process mesh in parts for very large models
- Ensure sufficient RAM for high resolutions

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.