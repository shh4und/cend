import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from skimage import img_as_float, exposure
from skimage.measure import marching_cubes
import open3d as o3d

def visualize_vessel_tree(vessel_tree, background_volume=None):
    """
    Visualiza a árvore vascular com ramos coloridos.

    Parameters:
    -----------
    vessel_tree : dict
        Dicionário contendo as centerlines e suas relações de parentesco
    background_volume : ndarray, optional
        Volume de fundo para exibir como referência (como máximo de intensidade)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Cria um mapa de cores para os diferentes ramos
    colors_list = list(plt.cm.tab20(np.linspace(0, 1, len(vessel_tree))))

    # Plot cada ramo da árvore
    for branch_id, branch_data in vessel_tree.items():
        centerline = branch_data["centerline"]
        parent_id = branch_data["parent"]

        # Extrai coordenadas z, y, x
        if len(centerline) > 0:
            # Converte todos os pontos para array numpy para garantir consistência
            points = np.array([np.array(p) for p in centerline])
            z, y, x = points[:, 0], points[:, 1], points[:, 2]

            # Define a cor com base no ID do ramo
            color = colors_list[branch_id % len(colors_list)]

            # Plota a centerline
            ax.plot(
                x,
                y,
                z,
                "-",
                linewidth=2,
                color=color,
                label=f"Branch {branch_id}"
                + (f" (parent: {parent_id})" if parent_id is not None else ""),
            )

            # Marca o início do ramo com um ponto maior
            ax.scatter(x[0], y[0], z[0], s=50, c=[color], marker="o")

    # Adiciona o volume de fundo se fornecido
    if background_volume is not None:
        # Cria uma visualização MIP (Maximum Intensity Projection)
        mip_z = np.max(background_volume, axis=0)
        extent_y = (0, background_volume.shape[1])
        extent_x = (0, background_volume.shape[2])

        # Plota o MIP como um plano na base
        z_min = ax.get_zlim()[0]
        mip_cmap = plt.cm.gray
        ax.contourf(
            np.arange(extent_x[0], extent_x[1]),
            np.arange(extent_y[0], extent_y[1]),
            mip_z,
            zdir="z",
            offset=z_min,
            cmap=mip_cmap,
            alpha=0.5,
        )

    # Configurações do gráfico
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Vessel Tree Visualization")

    # Adiciona legenda se houver múltiplos ramos
    if len(vessel_tree) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1))

    plt.tight_layout()
    return fig, ax


def quick_plot_tree(vessel_tree):
    """Visualização rápida da árvore vascular para notebook"""
    plt.figure(figsize=(10, 8))

    # Para cada ramo
    for branch_id, branch_data in vessel_tree.items():
        centerline = branch_data["centerline"]

        if len(centerline) > 0:
            # Extrai pontos
            points = np.array([np.array(p) for p in centerline])

            # Para visualização 2D, usamos apenas y e x
            plt.plot(
                points[:, 2],
                points[:, 1],
                "-o",
                linewidth=2,
                label=f"Branch {branch_id}",
            )

            # Marca o início do ramo
            plt.plot(points[0, 2], points[0, 1], "D", markersize=10)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Vessel Tree - 2D Projection")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def interactive_vessel_tree(centerline, background_volume=None, opacity=0.7):
    """
    Cria uma visualização 3D interativa da árvore vascular usando Plotly.

    Parameters:
    -----------
    vessel_tree : dict
        Dicionário contendo as centerlines e suas relações de parentesco
    background_volume : ndarray, optional
        Volume de fundo para exibir como referência
    opacity : float, optional
        Opacidade para o volume de fundo

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Figura interativa que pode ser exibida no notebook ou salva como HTML
    """
    # Inicializa a figura
    fig = go.Figure()

    # Paleta de cores
    colorscale = px.colors.qualitative.Plotly


    if len(centerline) > 0:
        # Converte para array numpy
        points = np.array([np.array(p) for p in centerline])
        z, y, x = points[:, 0], points[:, 1], points[:, 2]

        # Define a cor para este ramo
        color = colorscale[0 % len(colorscale)]

        # Adiciona a linha da centerline
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines+markers",
                marker=dict(size=4, color=color),
                line=dict(color=color, width=5),
                name=f"Branch {0}",
                #+ (f" (parent: {parent_id})" if parent_id is not None else ""),
            )
        )

        # Adiciona um marcador maior para o ponto inicial
        fig.add_trace(
            go.Scatter3d(
                x=x[0:1],
                y=y[0:1],
                z=z[0:1],
                mode="markers",
                marker=dict(size=8, color=color, symbol="diamond"),
                showlegend=False,
            )
        )

    # Adiciona o volume como isosuperfície se fornecido
    if background_volume is not None:
        # Para volumes grandes, você pode querer reduzir a resolução

        # Podemos usar um threshold para criar a isosuperfície
        threshold = np.percentile(background_volume, 80)  # Ajuste conforme necessário
        verts, faces, _, _ = marching_cubes(background_volume, threshold)

        x, y, z = verts[:, 2], verts[:, 1], verts[:, 0]

        i, j, k = faces.T
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                opacity=opacity,
                color="lightgray",
                name="Volume",
            )
        )

    # Layout e configurações da câmera
    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="3D Vessel Tree Visualization",
    )

    return fig


def interactive_volume_view(background_volume, opacity=0.7, threshold=None, color='lightgray'):
    """
    Cria uma visualização 3D interativa de um volume usando Plotly.

    Parameters:
    -----------
    background_volume : ndarray
        Volume 3D para visualização
    opacity : float, optional
        Opacidade para o volume (0.0-1.0)
        Default: 0.7
    threshold : float, optional
        Valor de threshold para a isosuperfície. Se None, usa o percentil 80
        Default: None
    color : str, optional
        Cor da isosuperfície
        Default: 'lightgray'

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Figura interativa que pode ser exibida no notebook ou salva como HTML
    """
    # Inicializa a figura
    fig = go.Figure()

    # Cria a isosuperfície do volume usando marching cubes
    
    # Define o threshold para a isosuperfície
    if threshold is None:
        threshold = np.percentile(background_volume, 80)
        
    # Extrai os vértices e faces da isosuperfície
    verts, faces, _, _ = marching_cubes(background_volume, threshold)
    
    # Ajusta as coordenadas
    x, y, z = verts[:, 2], verts[:, 1], verts[:, 0]
    
    # Adiciona a mesh 3D
    i, j, k = faces.T
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            opacity=opacity,
            color=color,
            name='Volume',
            showscale=False
        )
    )

    # Layout e configurações da câmera
    fig.update_layout(
        scene=dict(
            xaxis_title="X", 
            yaxis_title="Y", 
            zaxis_title="Z", 
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="3D Volume Visualization",
    )

    return fig

def interactive_multithreshold_volume(volume, thresholds=None, colors=None, opacities=None):
    """
    Visualiza um volume com múltiplas isosuperfícies em diferentes thresholds.
    
    Parameters:
    -----------
    volume : ndarray
        Volume 3D para visualização
    thresholds : list of float, optional
        Lista de valores de threshold para as isosuperfícies
    colors : list of str, optional
        Lista de cores para cada isosuperfície
    opacities : list of float, optional
        Lista de valores de opacidade para cada isosuperfície
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Figura interativa
    """
    # Valores padrão
    if thresholds is None:
        # Valores percentis para diferentes estruturas
        thresholds = [
            np.percentile(volume, 60),  # Estruturas menos densas
            np.percentile(volume, 80),  # Estruturas médias
            np.percentile(volume, 90)   # Estruturas mais densas
        ]
    
    if colors is None:
        colors = ['lightblue', 'lightgray', 'white']
    
    if opacities is None:
        opacities = [0.3, 0.5, 0.7]
    
    # Garante que as listas tenham o mesmo tamanho
    n = min(len(thresholds), len(colors), len(opacities))
    
    fig = go.Figure()
    
    # Para cada threshold, cria uma isosuperfície
    for i in range(n):
        verts, faces, _, _ = marching_cubes(volume, thresholds[i])
        
        x, y, z = verts[:, 2], verts[:, 1], verts[:, 0]
        i_faces, j_faces, k_faces = faces.T
        
        fig.add_trace(
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i_faces, j=j_faces, k=k_faces,
                opacity=opacities[i],
                color=colors[i],
                name=f'Threshold {thresholds[i]:.2f}'
            )
        )
    
    fig.update_layout(
        scene=dict(
            xaxis_title="X", 
            yaxis_title="Y", 
            zaxis_title="Z", 
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="Multi-threshold Volume Visualization"
    )
    
    return fig

def mpl_projection2d(
    volume,
    method="max",
    axis=0,
    figsize=(14, 10),
    cmap="gray",
    title=None,
    save_path=None,
    normalize=True,
):
    """
    Cria uma projeção 2D de um volume 3D utilizando matplotlib.

    Parameters
    ----------
    volume : ndarray
        Volume 3D de entrada
    method : str, optional
        Método de projeção: 'max' (máximo), 'mean' (média), 'sum' (soma)
        Default: 'max'
    axis : int, optional
        Eixo ao longo do qual fazer a projeção (0, 1, ou 2)
        Default: 0
    figsize : tuple, optional
        Tamanho da figura (largura, altura) em polegadas
        Default: (10, 8)
    cmap : str, optional
        Mapa de cores do matplotlib
        Default: 'viridis'
    title : str, optional
        Título do gráfico
        Default: None
    save_path : str, optional
        Caminho para salvar a figura. Se None, a figura não é salva.
        Default: None
    normalize : bool, optional
        Se True, normaliza a projeção para [0, 1]
        Default: True

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura do matplotlib
    ax : matplotlib.axes.Axes
        Eixos do matplotlib com a projeção
    projection : ndarray
        Array 2D com os valores da projeção
    """
    # Certificar que o volume é float para operações
    volume = img_as_float(volume)

    # Calcular projeção baseada no método selecionado
    if method == "max":
        projection = np.max(volume, axis=axis)
    elif method == "min":
        projection = np.min(volume, axis=axis)
    elif method == "mean":
        projection = np.mean(volume, axis=axis)
    elif method == "sum":
        projection = np.sum(volume, axis=axis)
    else:
        raise ValueError("Método deve ser 'max', 'min', 'mean' ou 'sum'")

    # Normalizar se necessário
    if normalize:
        projection = exposure.rescale_intensity(projection, out_range=(0, 1))

    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)

    # Plotar a projeção
    im = ax.imshow(projection, cmap=cmap)

    # Adicionar barra de cores
    plt.colorbar(im, ax=ax, label="Intensidade")

    # Adicionar título se fornecido
    if title:
        ax.set_title(title)

    # Remover ticks dos eixos para uma aparência mais limpa
    ax.set_xticks([])
    ax.set_yticks([])

    # Ajustar layout
    plt.tight_layout()

    # Salvar se um caminho foi fornecido
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax, projection


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