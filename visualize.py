import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from skimage import img_as_float, exposure
from skimage.measure import marching_cubes
import open3d as o3d
import logging

logging.basicConfig(level=logging.INFO)

def o3d_interactive_vessel_tree(vessel_tree, background_volume=None, opacity=0.7):
    """
    Create an interactive 3D visualization of the vessel tree using Open3D.
    
    Parameters:
    -----------
    vessel_tree : dict
        Dictionary containing the centerlines and their parent-child relationships
    background_volume : ndarray, optional
        Background volume to display as reference
    opacity : float, optional
        Opacity for the background volume (0.0-1.0)
        Default: 0.7
        
    Returns:
    --------
    None (displays interactive visualization window)
    """
    
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Vessel Tree Visualization", width=1024, height=768)
    
    # Create a colormap for branches
    colors_list = list(plt.cm.tab20(np.linspace(0, 1, len(vessel_tree))))
    
    # Create line sets for each branch
    for branch_id, branch_data in vessel_tree.items():
        centerline = branch_data["centerline"]
        parent_id = branch_data["parent"]
        
        if len(centerline) > 0:
            # Convert centerline to numpy array
            points = np.array([np.array(p) for p in centerline])
            
            # Create a line set for the centerline
            line_set = o3d.geometry.LineSet()
            
            # Set points
            line_set.points = o3d.utility.Vector3dVector(points)
            
            # Create lines (connections between consecutive points)
            lines = [[i, i+1] for i in range(len(points)-1)]
            line_set.lines = o3d.utility.Vector2iVector(lines)
            
            # Set color
            color = colors_list[branch_id % len(colors_list)][:3]  # RGB only
            line_colors = [color for _ in range(len(lines))]
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            
            # Add to visualizer
            vis.add_geometry(line_set)
            
            # Create a sphere for the start point of the branch
            start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
            start_sphere.translate(points[0])
            start_sphere.paint_uniform_color(color)
            vis.add_geometry(start_sphere)
    
    # Add the background volume as an isosurface if provided
    if background_volume is not None:
        # Create isosurface using marching cubes
        threshold = np.percentile(background_volume, 80)
        verts, faces, _, _ = marching_cubes(background_volume, threshold)
        
        # Create a mesh for the isosurface
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Set color and opacity
        mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
        mesh.compute_vertex_normals()
        
        # Add to visualizer
        vis.add_geometry(mesh)
    
    # Setup visualization properties
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 5.0
    opt.line_width = 2.0
    
    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    vis.add_geometry(coord_frame)
    
    # Setup view
    vis.get_view_control().set_zoom(0.8)
    
    # Add key callbacks for better interaction
    # def rotate_view(vis):
    #     ctr = vis.get_view_control()
    #     ctr.rotate(10.0, 0.0)
    #     return False
    
    # Run the visualizer
    # vis.register_animation_callback(rotate_view)
    vis.run()
    vis.destroy_window()


def o3d_interactive_centerline(centerline, background_volume=None, opacity=0.7):
    """
    Create an interactive 3D visualization of a single centerline using Open3D.
    
    Parameters:
    -----------
    centerline : list
        List of points defining the centerline
    background_volume : ndarray, optional
        Background volume to display as reference
    opacity : float, optional
        Opacity for the background volume (0.0-1.0)
        Default: 0.7
        
    Returns:
    --------
    None (displays interactive visualization window)
    """
    
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Centerline Visualization", width=1024, height=768)
    
    if len(centerline) > 0:
        # Convert centerline to numpy array
        points = np.array([np.array(p) for p in centerline])
        
        # Create a line set for the centerline
        line_set = o3d.geometry.LineSet()
        
        # Set points
        line_set.points = o3d.utility.Vector3dVector(points)
        
        # Create lines (connections between consecutive points)
        lines = [[i, i+1] for i in range(len(points)-1)]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Set color (blue)
        color = [0.0, 0.6, 1.0]  # Blue color
        line_colors = [color for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        
        # Add to visualizer
        vis.add_geometry(line_set)
        
        # Create spheres for the start and end points
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
        start_sphere.translate(points[0])
        start_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Green for start
        vis.add_geometry(start_sphere)
        
        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
        end_sphere.translate(points[-1])
        end_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red for end
        vis.add_geometry(end_sphere)
    
    # Add the background volume as an isosurface if provided
    if background_volume is not None:
        # Create isosurface using marching cubes
        threshold = np.percentile(background_volume, 80)  # Adjust as needed
        verts, faces, _, _ = marching_cubes(background_volume, threshold)
        
        # Create a mesh for the isosurface
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Set color and opacity
        mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
        mesh.compute_vertex_normals()
        
        # Add to visualizer
        vis.add_geometry(mesh)
    
    # Setup visualization properties
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 5.0
    opt.line_width = 2.0
    
    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    vis.add_geometry(coord_frame)
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()


def o3d_interactive_maxima_points(maxima_coords, background_volume=None, point_color=[1.0, 0.1, 0.1], point_size=2.0, threshold_percentile=80):
    """
    Cria uma visualização 3D interativa para os pontos de máxima local (ex: do VFC).
    Versão corrigida para ser compatível com versões mais recentes do Open3D.

    Parâmetros:
    -----------
    maxima_coords : ndarray
        Array (Nx3) de coordenadas (z, y, x) dos pontos de máxima.
    background_volume : ndarray, opcional
        Volume 3D de fundo para exibir como referência contextual.
    point_color : list, opcional
        Cor [R, G, B] para os pontos de máxima. Padrão é vermelho.
    point_size : float, opcional
        Tamanho dos pontos na visualização.
    threshold_percentile : int, opcional
        Percentil do limiar para gerar a malha do volume de fundo.
        
    Retorna:
    --------
    None (exibe uma janela de visualização interativa).
    """
    
    # Criar uma janela de visualização
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Visualização 3D dos Pontos de Máxima", width=1280, height=800)
    
    # Adicionar os pontos de máxima como uma nuvem de pontos (PointCloud)
    if maxima_coords is not None and len(maxima_coords) > 0:
        # Criar o objeto PointCloud
        point_cloud = o3d.geometry.PointCloud()
        
        # Open3D espera coordenadas (x, y, z), mas seus dados estão em (z, y, x).
        # Invertemos as colunas para a visualização correta.
        coords_for_o3d = maxima_coords[:, ::-1] # Inverte a ordem das colunas para (x, y, z)
        
        # Definir os pontos
        point_cloud.points = o3d.utility.Vector3dVector(maxima_coords)
        
        # Definir a cor
        point_cloud.paint_uniform_color(point_color)
        
        # Adicionar a nuvem de pontos ao visualizador
        vis.add_geometry(point_cloud)
        logging.info(f"Visualizando {len(maxima_coords)} pontos de máxima.")

    # Adicionar o volume de fundo como uma malha (isosuperfície), se fornecido
    if background_volume is not None:
        logging.info("Gerando malha do volume de fundo...")
        # Criar isosuperfície usando marching cubes
        threshold = np.percentile(background_volume, threshold_percentile)
        verts, faces, _, _ = marching_cubes(background_volume, threshold)
        
        # Open3D espera coordenadas (x, y, z)
        verts_for_o3d = verts[:, ::-1] # Inverte (z,y,x) para (x,y,z)

        # Criar a malha para a isosuperfície
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Definir cor e calcular normais para sombreamento adequado
        mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Cinza claro
        mesh.compute_vertex_normals()
        
        # Adicionar a malha ao visualizador
        vis.add_geometry(mesh)
        logging.info(f"Malha do volume de fundo adicionada com limiar no percentil {threshold_percentile}.")

    
    # Configurar propriedades de renderização
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.15, 0.15, 0.15])  # Fundo cinza escuro
    opt.point_size = point_size
    # As linhas que causaram o erro foram removidas. O Open3D usará o shader padrão.
    # mat = opt.material  <- REMOVIDA
    # mat.shader = 'defaultUnlit' <- REMOVIDA
    
    # Adicionar um sistema de coordenadas para referência
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0,0,0])
    vis.add_geometry(coord_frame)
    
    # Executar a janela de visualização interativa
    logging.info("\nFeche a janela do Open3D para continuar a execução do script.")
    vis.run()
    vis.destroy_window()
