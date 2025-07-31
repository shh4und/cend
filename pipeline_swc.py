# pipeline_swc.py (versão atualizada)
import sys
sys.path.append('~/RMNIM')
from ip.graph_nx import Graph # type: ignore

import argparse
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Tuple, Optional, List
import gc # Importa o Garbage Collector

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from tqdm import tqdm

from dst_fields import DistanceFields
from image_io import load_3d_volume
from vfc import create_maxima_image


def process_image(args: Tuple):
    """Executa o pipeline completo para uma única imagem com otimização de memória."""
    (
        img_idx, img_path, root_coord, output_dir, sigma_range,
        neuron_threshold, mov_avg, pruning_threshold
    ) = args

    try:
        logging.info(f"Processando imagem {img_idx+1}: {img_path.name}")
        
        # 1. Carregar imagem e converter para float32 para economizar memória
        volume = load_3d_volume(str(img_path)) #.astype(np.float32, copy=False)

        # 2. Pré-processamento
        gauss_filtered = ndi.gaussian_filter(volume, 2.0)
        min_filtered = ndi.minimum_filter(gauss_filtered, 2)
        volume[min_filtered<=0] = 0
        del gauss_filtered, min_filtered; gc.collect() # Limpeza

        # 3. Filtragem e Segmentação
        df = DistanceFields(
            volume=volume, sigma_range=sigma_range,
            neuron_threshold=neuron_threshold, seed_point=root_coord
        )
        # Otimização: Passar dtype para os métodos se eles suportarem
        img_filtered = df.multiscale_anisotropic()
        img_mask, _ = df.adaptive_mean_mask(img_filtered)
        del img_filtered; gc.collect() # Limpeza
        
        clean_img_mask = df.morphological_denoising(img_mask)
        pressure_field = ndi.gaussian_filter(df.pressure_field(clean_img_mask), 1.5)
        thrust_field = ndi.gaussian_filter(df.thrust_field(clean_img_mask), 1.0)
        del img_mask; gc.collect() # Limpeza

        # 4. Esqueletização
        maximas_set = df.find_thrust_maxima(thrust_field, clean_img_mask, order=3)
        skel_coords = df.generate_skel_from_seed(maximas_set, root_coord, pressure_field, clean_img_mask)
        skel_img = create_maxima_image(skel_coords, volume.shape)
        clean_skel = skeletonize(skel_img)
        del clean_img_mask, thrust_field, skel_img; gc.collect() # Limpeza

        # 5. Geração do Grafo
        g = Graph(clean_skel)
        g.create_graph(moving_avg=mov_avg)
        del clean_skel; gc.collect() # Limpeza
        
        # 6. Poda
        if pruning_threshold > 0:
            g.prune_by_branch_length(pruning_threshold)

        # 7. Validação da Raiz
        pruned_skel_img = np.zeros(volume.shape, dtype=np.uint8)
        if g.get_graph().number_of_nodes() > 0:
            nodes = np.array(list(g.get_graph().nodes())).astype(int)
            pruned_skel_img[nodes[:, 0], nodes[:, 1], nodes[:, 2]] = 1
        del volume # Não precisamos mais do volume original
        
        valid_root = df.correct_and_update_root(pruned_skel_img.astype(bool), root_coord)
        del df, pruned_skel_img; gc.collect() # Limpeza

        if valid_root is None:
            logging.error(f"Não foi possível determinar uma raiz válida para a imagem {img_idx+1}. Pulando.")
            return None

        # 8. Geração do SWC
        g.set_root(valid_root)
        labeled_mst = g.apply_dfs_and_label_nodes()
        output_filename = output_dir / f"OP_{img_idx+1}_reconstruction.swc"
        success = g.save_to_swc(labeled_mst, str(output_filename), pressure_field)
        del pressure_field; gc.collect()

        if success:
            logging.info(f"Imagem {img_idx+1} salva com sucesso em {output_filename}")
            return str(output_filename)
        else:
            logging.error(f"Falha ao salvar o arquivo SWC para a imagem {img_idx+1}.")
            return None
    except Exception as e:
        logging.error(f"Erro inesperado ao processar a imagem {img_idx+1}: {e}", exc_info=True)
        return None

# ... (O resto do script 'main' permanece o mesmo)
def main():
    # ... (código do parser e setup como antes)
    parser = argparse.ArgumentParser(
        description="Pipeline para reconstrução de neurônios e geração de arquivos SWC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ... (argumentos anteriores)
    parser.add_argument("--data_dir", type=str, default="data", help="Diretório contendo as pastas das imagens.")
    parser.add_argument("--output_dir", type=str, default="results_swc", help="Diretório para salvarp os arquivos SWC.")
    parser.add_argument("--image_index", type=int, default=None, help="Índice da imagem a ser processada (1 a 9). Se não, processa todas.")
    parser.add_argument("--parallel_jobs", type=int, default=2, help="Número de processos paralelos. Padrão: 2 para segurança de memória.")
    parser.add_argument("--sigma_min", type=float, default=3.0, help="Sigma mínimo.")
    parser.add_argument("--sigma_max", type=float, default=5.0, help="Sigma máximo.")
    parser.add_argument("--sigma_step", type=float, default=1.0, help="Passo do sigma.")
    parser.add_argument("--neuron_threshold", type=float, default=0.05, help="Threshold de tubularidade.")
    parser.add_argument("--moving_avg", type=bool, default=False, help="Interpolacao por media movel")

    
    # Novo argumento para poda
    parser.add_argument("--pruning_threshold", type=int, default=15, help="Comprimento máximo (em pixels/nós) de um ramo para ser podado. Defina como 0 para desativar a poda.")

    args = parser.parse_args()
    
    # ... (resto do código main, como antes)
    # A única mudança é adicionar 'args.pruning_threshold' ao tuple 'tasks'
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted([p for p in data_dir.glob("OP_*") if p.is_dir()])
    roots = [(0, 429, 31), (25, 391, 1), (38, 179, 94), (0, 504, 128), (33, 264, 185), (10, 412, 15), (39, 216, 120), (55, 181, 119), (4, 364, 64)]

    if not image_paths or len(image_paths) != len(roots):
        logging.error("Erro: Diretórios de imagem não encontrados ou número de raízes incompatível.")
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
        tasks.append((i, image_paths[i], roots[i], output_dir, sigma_range, args.neuron_threshold, args.moving_avg, args.pruning_threshold))

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
        logging.info(f"Executando {len(tasks)} tarefa(s) em paralelo com {num_jobs} jobs.")
        with Pool(processes=num_jobs) as pool:
            list(tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks), desc="Processando Imagens"))
            
    logging.info("Pipeline concluído.")

if __name__ == "__main__":
    main()