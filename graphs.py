import networkx as nx
import numpy as np
from swc import * # Assumindo que a classe SWCFile existe como no seu código
from typing import Tuple, Optional, List
from collections import deque
from scipy.interpolate import splprep, splev
import scipy.ndimage as ndi

class Graph:
    """
    Uma classe refatorada para criar, processar e salvar um grafo de uma imagem 3D.

    O fluxo de trabalho principal é:
    1. Inicializar com uma imagem e um voxel raiz.
    2. Criar o grafo de forma eficiente a partir da raiz.
    3. Calcular a Árvore Geradora Mínima (MST).
    4. Opcionalmente, podar ramos curtos da MST.
    5. Rotular os nós da MST para o formato SWC.
    6. Salvar o resultado em um arquivo .swc.
    """
    def __init__(self, image: np.ndarray, root_voxel: Tuple[int, int, int]):
        """
        Inicializa o objeto Graph.

        Args:
            image (np.ndarray): A imagem 3D (binária ou em tons de cinza).
            root_voxel (Tuple[int, int, int]): As coordenadas (z, y, x) do nó raiz.
        """
        if not (0 <= root_voxel[0] < image.shape[0] and
                0 <= root_voxel[1] < image.shape[1] and
                0 <= root_voxel[2] < image.shape[2] and
                image[root_voxel] != 0):
            raise ValueError("O voxel raiz está fora dos limites da imagem ou tem valor zero.")

        self.image = image
        self.shape = image.shape
        self.root = root_voxel
        
        self.graph = nx.Graph()
        self.mst: Optional[nx.Graph] = None
        
        print(">> Iniciando a criação do grafo a partir da raiz...")
        self._create_graph_from_root()

    def _create_graph_from_root(self) -> None:
        """
        Cria o grafo de forma eficiente usando uma busca em largura (BFS) a partir da raiz.
        Isso evita a criação de um grafo denso de toda a imagem, focando apenas
        no componente conectado à raiz.
        """
        queue = deque([self.root])
        visited = {self.root}
        self.graph.add_node(self.root, pos=self.root)

        while queue:
            current_voxel = queue.popleft()
            
            # Busca os 26 vizinhos do voxel atual
            for neighbor in self._get_26_neighborhood(current_voxel):
                if neighbor not in visited:
                    visited.add(neighbor)
                    
                    # Adiciona o nó e a aresta com seu peso euclidiano
                    self.graph.add_node(neighbor, pos=neighbor)
                    weight = self._euclidean_distance(current_voxel, neighbor)
                    self.graph.add_edge(current_voxel, neighbor, weight=weight)
                    
                    queue.append(neighbor)
        
        print(f">> Grafo criado com {self.graph.number_of_nodes()} nós e {self.graph.number_of_edges()} arestas.")

    def calculate_mst(self) -> None:
        """Calcula a Árvore Geradora Mínima (MST) e a armazena no atributo self.mst."""
        print(">> Calculando a Árvore Geradora Mínima (MST)...")
        if not self.graph:
            print("Aviso: O grafo está vazio. A MST não pode ser calculada.")
            return
            
        self.mst = nx.minimum_spanning_tree(self.graph, weight="weight", algorithm="kruskal")
        print(">> MST calculada.")

    def prune_mst_by_length(self, length_threshold: int):
        """
        Poda ramos curtos da MST. A poda é muito mais eficiente na MST do que no grafo completo.

        Args:
            length_threshold (int): O comprimento máximo (em número de nós) para um ramo ser podado.
        """
        if not self.mst:
            print("Aviso: A MST precisa ser calculada antes da poda. Chame `calculate_mst()` primeiro.")
            return
        if length_threshold <= 0:
            return

        print(f">> Iniciando a poda de ramos da MST com comprimento <= {length_threshold}")
        
        # Trabalhar em uma cópia para evitar problemas de iteração
        mst_copy = self.mst.copy()
        
        # Encontra todos os pontos finais (folhas) na MST
        endpoints = [node for node, degree in mst_copy.degree() if degree == 1]
        nodes_to_remove = set()

        for start_node in endpoints:
            if start_node in nodes_to_remove:
                continue

            path = [start_node]
            current_node = start_node
            
            # Percorre o ramo a partir da folha até encontrar uma junção (grau > 2) ou o fim
            while mst_copy.degree(current_node) < 3:
                # Em uma árvore, um nó de grau 2 tem exatamente um vizinho não visitado no caminho
                neighbors = [n for n in mst_copy.neighbors(current_node) if n not in path]
                if not neighbors:
                    break # Chegou ao fim de um fragmento isolado
                
                current_node = neighbors[0]
                path.append(current_node)

                # Para o loop de segurança se o caminho ficar muito longo
                if len(path) > (self.graph.number_of_nodes()): break
            
            # O último nó no 'path' é o ponto de junção. O ramo a ser podado não o inclui.
            branch_to_prune = path[:-1]
            if len(branch_to_prune) <= length_threshold:
                nodes_to_remove.update(branch_to_prune)

        if nodes_to_remove:
            self.mst.remove_nodes_from(list(nodes_to_remove))
            print(f">> Poda da MST concluída. {len(nodes_to_remove)} nós removidos.")
            
    def validate_and_set_root(self, original_root: Tuple[int, int, int]) -> None:
        """
        Verifica se a raiz original é um nó do grafo. Se não for, encontra o nó
        mais próximo e atualiza self.root.

        Args:
            original_root (Tuple[int, int, int]): A coordenada da raiz original.
        """
        # Se a raiz já é um nó válido do grafo, não há nada a fazer.
        if self.graph.has_node(original_root):
            self.root = original_root
            print(f">> Raiz original {original_root} é válida e está no grafo.")
            return

        print(f"Aviso: Raiz original {original_root} não encontrada no esqueleto. Procurando o nó mais próximo...")
        
        nodes = np.array(list(self.graph.nodes()))
        # Calcula a distância euclidiana ao quadrado (mais rápido) de todos os nós para a raiz original
        distances_sq = np.sum((nodes - np.array(original_root))**2, axis=1)
        
        # Encontra o índice do nó com a menor distância
        closest_node_index = np.argmin(distances_sq)
        new_root = tuple(nodes[closest_node_index])
        
        self.root = new_root
        print(f">> Raiz atualizada para o nó mais próximo: {self.root}")


    def label_nodes_for_swc(self) -> None:
        """
        Aplica uma busca em profundidade (DFS) na MST para rotular os nós
        com 'id' e 'parent_id', preparando para o formato SWC.
        """
        if not self.mst:
            print("Aviso: A MST precisa ser calculada. Chame `calculate_mst()` primeiro.")
            return
            
        print(">> Iniciando rotulagem dos nós via DFS...")
        visited = set()
        stack = [(self.root, -1)]  # A pilha contém (voxel, parent_id)
        node_id_counter = 1

        while stack:
            voxel, parent_id = stack.pop()
            if voxel not in visited:
                visited.add(voxel)
                
                # Garante que o nó ainda existe na MST após a poda
                if self.mst.has_node(voxel):
                    if "id" not in self.mst.nodes[voxel]:
                        self.mst.nodes[voxel]["id"] = node_id_counter
                        node_id_counter += 1
                    
                    self.mst.nodes[voxel]["parent"] = parent_id
                    
                    # Adiciona os vizinhos à pilha
                    current_node_id = self.mst.nodes[voxel]["id"]
                    # Iterar em uma cópia dos vizinhos para evitar problemas se o grafo for modificado
                    for neighbor in list(self.mst.neighbors(voxel)):
                        if neighbor not in visited:
                            stack.append((neighbor, current_node_id))
        
        print(">> Rotulagem concluída.")

    # É necessário adicionar esta importação no topo do seu arquivo graph_nx2.py

    # In graph_nx2.py

    def generate_smoothed_swc(
        self, 
        filename: str, 
        pressure_field: Optional[np.ndarray] = None, 
        smoothing_factor: float = 0.5, 
        num_points_per_branch: int = 20
    ) -> bool:
        """
        Decompõe a MST em ramos, suaviza cada um com splines e salva em um arquivo SWC.
        (Versão corrigida com lógica de ID robusta)
        """
        if not self.mst:
            print("Aviso: A MST precisa ser calculada primeiro.")
            return False

        print(">> Iniciando suavização por splines e geração de SWC...")
        swc = SWCFile(filename)
        
        # Dicionário para mapear o voxel original para o novo ID do SWC.
        # A raiz sempre tem o ID 1 e pai -1.
        voxel_to_id_map = {self.root: 1}
        node_id_counter = 2 # Começa em 2 porque a raiz é 1.
        
        # Adiciona o ponto raiz ao SWC
        z_root, y_root, x_root = self.root
        radius = 1.0
        if pressure_field is not None:
            radius = max(1.0, float(pressure_field[int(z_root), int(y_root), int(x_root)]))
        swc.add_point(1, 9, x_root, y_root, z_root, radius, -1)

        visited_edges = set()

        # Itera sobre a árvore a partir da raiz para manter a ordem pai-filho
        for start_node in nx.dfs_preorder_nodes(self.mst, source=self.root):
            for neighbor in self.mst.neighbors(start_node):
                edge = tuple(sorted((start_node, neighbor)))
                if edge in visited_edges:
                    continue

                # 1. Encontrar o ramo completo
                path = [start_node, neighbor]
                curr = neighbor
                while self.mst.degree(curr) == 2:
                    next_node = [n for n in self.mst.neighbors(curr) if n != path[-2]][0]
                    path.append(next_node)
                    curr = next_node
                
                for i in range(len(path) - 1):
                    visited_edges.add(tuple(sorted((path[i], path[i+1]))))

                # 2. Suavizar o ramo com spline
                path_coords = np.array(path).T
                if len(path) < 4:
                    smoothed_points = np.array(path)
                else:
                    tck, u = splprep(path_coords, s=smoothing_factor, k=3)
                    u_new = np.linspace(u.min(), u.max(), num_points_per_branch)
                    z_new, y_new, x_new = splev(u_new, tck)
                    smoothed_points = np.vstack([z_new, y_new, x_new]).T

                # 3. Adicionar os pontos suavizados ao SWC com IDs crescentes
                # O pai do primeiro ponto do ramo é o nó de partida
                parent_id = voxel_to_id_map[start_node]
                
                # Pula o primeiro ponto do 'smoothed_points' porque ele corresponde
                # ao 'start_node' que já está (ou será) no SWC.
                for i, point_coords in enumerate(smoothed_points[1:]):
                    current_id = node_id_counter
                    z, y, x = point_coords
                    
                    radius = 1.0
                    if pressure_field is not None:
                        # Usamos `ndi.map_coordinates` para uma interpolação suave do raio
                        radius_val = ndi.map_coordinates(pressure_field, [[z], [y], [x]], order=1, mode='nearest')[0]
                        radius = max(1.0, float(radius_val))

                    swc.add_point(current_id, 9, x, y, z, radius, parent_id)
                    
                    # Atualiza o parent_id para o próximo ponto na sequência
                    parent_id = current_id
                    node_id_counter += 1

                # Atualiza o mapa com o ID do último ponto do ramo.
                # `parent_id` agora contém o ID do último ponto que foi adicionado.
                voxel_to_id_map[path[-1]] = parent_id

        return swc.write_file()
    
    def save_to_swc(self, filename: str, pressure_field: Optional[np.ndarray] = None) -> bool:
        """
        Salva a MST rotulada em um arquivo no formato SWC.
        
        Args:
            filename (str): O nome do arquivo de saída (ex: 'neuron.swc').
            pressure_field (np.ndarray, optional): Um campo 3D com informações de raio/largura.
        
        Returns:
            bool: True se o arquivo foi salvo com sucesso.
        """
        if not self.mst:
            print("Aviso: Não há MST para salvar.")
            return False

        swc = SWCFile(filename)
        
        # Ordena os nós pelo seu ID para um arquivo SWC bem formatado
        nodes_sorted = sorted(self.mst.nodes(data=True), key=lambda x: x[1].get('id', float('inf')))

        for node, attrs in nodes_sorted:
            if "id" not in attrs:
                continue
            
            z, y, x = map(float, node)
            radius = 1.0  # Raio padrão

            if pressure_field is not None and isinstance(pressure_field, np.ndarray):
                iz, iy, ix = int(z), int(y), int(x)
                if (0 <= iz < pressure_field.shape[0] and
                    0 <= iy < pressure_field.shape[1] and
                    0 <= ix < pressure_field.shape[2] and
                    pressure_field[iz, iy, ix] > 0):
                    radius = pressure_field[iz, iy, ix]

            swc.add_point(attrs["id"], 9, x, y, z, radius, attrs.get("parent", -1))

        print(f">> Salvando MST em {filename}")
        return swc.write_file()

    # --- MÉTODOS AUXILIARES ---
    def _get_26_neighborhood(self, voxel: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        z, y, x = voxel
        neighbors = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if (0 <= nz < self.shape[0] and 
                        0 <= ny < self.shape[1] and 
                        0 <= nx < self.shape[2] and 
                        self.image[nz, ny, nx] != 0):
                        neighbors.append((nz, ny, nx))
        return neighbors

    @staticmethod
    def _euclidean_distance(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
        return np.linalg.norm(np.array(p1) - np.array(p2))