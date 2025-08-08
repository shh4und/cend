import argparse
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Tuple, Optional, List
import gc

import numpy as np
import networkx as nx
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from tqdm import tqdm

from dst_fields import DistanceFields
from image_io import load_3d_volume
from vfc import create_maxima_image
from graphs import Graph


def process_image(args: Tuple):
    """Executa o pipeline completo para uma única imagem com a classe de grafo refatorada."""
    (
        img_idx,
        img_path,
        root_coord,
        output_dir,
        sigma_range,
        neuron_threshold,
        pruning_threshold,
        maximas_min_dist,
        smoothing_factor,
        num_points_per_branch
    ) = args

    logging.info(f"Processando imagem {img_idx+1}: {img_path.name}")

    # 1. Carregar imagem
    volume = load_3d_volume(str(img_path))

    # 2. Pré-processamento
    gauss_filtered = ndi.gaussian_filter(volume, 1.0)
    min_filtered = ndi.minimum_filter(gauss_filtered, 2)
    volume[min_filtered <= 0] = 0
    del gauss_filtered, min_filtered; gc.collect()

    # 3. Filtragem e Segmentação
    df = DistanceFields(
        volume=volume,
        sigma_range=sigma_range,
        neuron_threshold=neuron_threshold,
        seed_point=root_coord,
        dataset_number=img_idx+1
    )
    img_filtered = df.multiscale_anisotropic()
    img_mask = df.adaptive_mean_mask(img_filtered)[0]
    del img_filtered; gc.collect()

    clean_img_mask = df.morphological_denoising(img_mask)
    pressure_field = ndi.gaussian_filter(df.pressure_field(clean_img_mask), 2.0)
    thrust_field = ndi.gaussian_filter(df.thrust_field(clean_img_mask), 1.0)
    del img_mask; gc.collect()

    # 4. Esqueletonização
    maximas_set = df.find_thrust_maxima(thrust_field, clean_img_mask, order=maximas_min_dist)
    skel_coords = df.generate_skel_from_seed(
        maximas_set, root_coord, pressure_field, clean_img_mask
    )
    skel_img = create_maxima_image(skel_coords, volume.shape)
    clean_skel = skeletonize(skel_img)
    del clean_img_mask, thrust_field, skel_img, skel_coords, maximas_set, df, volume; gc.collect()

    if not np.any(clean_skel):
        logging.error(f"Esqueleto vazio para a imagem {img_idx+1}. Pulando.")
        return None

    
    skel_points = np.argwhere(clean_skel)
    distances_sq = np.sum((skel_points - np.array(root_coord))**2)
    initial_valid_root = tuple(skel_points[np.argmin(distances_sq)])

    g = Graph(clean_skel, initial_valid_root)
    del clean_skel; gc.collect()

    g.calculate_mst()
    
    if pruning_threshold > 0:
        g.prune_mst_by_length(pruning_threshold)
        
        if not g.mst.has_node(g.root):
                logging.warning(f"A raiz {g.root} foi removida durante a poda. Encontrando uma nova raiz.")
                if g.mst.number_of_nodes() == 0:
                    logging.error("graph ficou vazia após a poda.")
                    return None
                
                main_component = max(nx.connected_components(g.mst), key=len)
                
                # Encontra o nó no componente principal mais próximo da raiz original
                nodes_in_component = np.array(list(main_component))
                distances_to_original_root_sq = np.sum((nodes_in_component - np.array(root_coord))**2)
                new_root = tuple(nodes_in_component[np.argmin(distances_to_original_root_sq)])
                
                g.root = new_root
                logging.info(f"Nova raiz definida como {new_root}")

    
    # 5.5. SUAVIZA A ÁRVORE E SALVA O ARQUIVO SWC
    # Esta chamada substitui g.label_nodes_for_swc() e g.save_to_swc()
    output_filename = output_dir / f"OP_{img_idx+1}_reconstruction.swc"
    success = g.generate_smoothed_swc(
        str(output_filename), 
        pressure_field,
        smoothing_factor=smoothing_factor, 
        num_points_per_branch=num_points_per_branch,
    )
    del pressure_field, g; gc.collect()

    if success:
        logging.info(f"Imagem {img_idx+1} salva com sucesso em {output_filename}")
        
        # --- INÍCIO DA MODIFICAÇÃO: SALVAR ARQUIVO DE METADADOS ---
        meta_filename = output_filename.with_suffix('.meta')
        try:
            with open(meta_filename, 'w') as meta_file:
                
                meta_file.write(f"sigma_range: [{sigma_range[0]}, {sigma_range[1]}, {sigma_range[2]}]\n")
                meta_file.write(f"neuron_threshold: {neuron_threshold}\n")
                meta_file.write(f"pruning_threshold: {pruning_threshold}\n")
                # Adicione quaisquer outros parâmetros que desejar salvar
            logging.info(f"Metadados salvos em {meta_filename}")
        except Exception as e:
            logging.error(f"Falha ao salvar arquivo de metadados: {e}")
        # --- FIM DA MODIFICAÇÃO ---
            
        return str(output_filename)
    else:
        logging.error(f"Falha ao salvar o arquivo SWC para a imagem {img_idx+1}.")
        return None

def main():
    
    parser = argparse.ArgumentParser(
        description="Pipeline para reconstrução de neurônios e geração de arquivos SWC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Diretório contendo as pastas das imagens.")
    parser.add_argument("--output_dir", type=str, default="results_swc", help="Diretório para salvarp os arquivos SWC.")
    parser.add_argument("--image_index", type=int, default=None, help="Índice da imagem a ser processada (1 a 9). Se não, processa todas.")
    parser.add_argument("--parallel_jobs", type=int, default=2, help="Número de processos paralelos. Padrão: 2 para segurança de memória.")
    parser.add_argument("--sigma_min", type=float, default=1.0, help="Sigma mínimo.")
    parser.add_argument("--sigma_max", type=float, default=2.0, help="Sigma máximo.")
    parser.add_argument("--sigma_step", type=float, default=0.5, help="Passo do sigma.")
    parser.add_argument("--neuron_threshold", type=float, default=0.05, help="Threshold de tubularidade.")
    parser.add_argument("--pruning_threshold", type=int, default=0, help="Comprimento máximo (em pixels/nós) de um ramo para ser podado. Defina como 0 para desativar a poda.")
    parser.add_argument("--maximas_min_dist", type=int, default=2, help="Tamanho da janela de maximas")
    parser.add_argument("--smoothing_factor", type=float, default=0.8, help="Fator de suavidade para a spline")
    parser.add_argument("--num_points_per_branch", type=int, default=15, help="Numero de pontos por ramo para suavização")

    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted([p for p in data_dir.glob("OP_*") if p.is_dir()])
    roots = [
        (0, 429, 31),
        (25, 391, 1),
        (38, 179, 94),
        (0, 504, 128),
        (33, 264, 185),
        (10, 412, 15),
        (39, 216, 120),
        (55, 181, 119),
        (4, 364, 64),
    ]

    if not image_paths or len(image_paths) != len(roots):
        logging.error(
            "Erro: Diretórios de imagem não encontrados ou número de raízes incompatível."
        )
        return

    sigma_range = (args.sigma_min, args.sigma_max, args.sigma_step)

    tasks = []
    task_indices = range(len(image_paths))
    if args.image_index is not None:
        if not (1 <= args.image_index <= len(image_paths)):
            logging.error(f"Índice de imagem inválido: {args.image_index}.")
            return
        task_indices = [args.image_index - 1]

    for i in task_indices:
        tasks.append(
            (
                i,
                image_paths[i],
                roots[i],
                output_dir,
                sigma_range,
                args.neuron_threshold,
                args.pruning_threshold,
                args.maximas_min_dist,
                args.smoothing_factor,
                args.num_points_per_branch
            )
        )

    if not tasks:
        logging.warning("Nenhuma tarefa para executar.")
        return

    # Ajuste para garantir que não usamos mais jobs que tarefas
    num_jobs = min(args.parallel_jobs, len(tasks))

    if num_jobs <= 1:
        # Execução sequencial
        logging.info(f"Executando {len(tasks)} tarefa(s) sequencialmente.")
        for task in tqdm(tasks, desc="Processando Imagens"):
            process_image(task)
    else:
        # Execução paralela
        logging.info(
            f"Executando {len(tasks)} tarefa(s) em paralelo com {num_jobs} jobs."
        )
        with Pool(processes=num_jobs) as pool:
            list(
                tqdm(
                    pool.imap_unordered(process_image, tasks),
                    total=len(tasks),
                    desc="Processando Imagens",
                )
            )

    logging.info("Pipeline concluído.")


if __name__ == "__main__":
    main()
