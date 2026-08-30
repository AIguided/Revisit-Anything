import func_vpr
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

import datetime
import time
import sys
# import utils
# import nbr_agg

import argparse
from place_rec_global_config import (
    datasets,
    experiments,
    list_image_names,
    resolve_dataset_paths,
    workdir_data,
)
from gt import get_gt


import utm
from glob import glob
from collections import defaultdict
import os
from os.path import join
from natsort import natsorted
import cv2
from typing import Literal, List
import torch
from tkinter import *
import matplotlib
from utilities import VLAD
from sklearn.decomposition import PCA
import pickle
import faiss
import json
from importlib import reload
from segvlad_vector_db import (
    artifact_paths,
    benchmark_index,
    build_flat_l2_index,
    build_image_offsets,
    load_ground_truth_csv,
    prepare_vectors,
    save_benchmark,
    save_database,
    save_queries,
    save_retrieval_csv,
)

# matplotlib.use('TkAgg')
matplotlib.use('Agg') #Headless

current_time = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")

# from sklearn.neighbors import NearestNeighbors
# from sklearn.neighbors import KDTree

def recall_segloc(
    workdir,
    dataset_name,
    experiment_config,
    experiment_name,
    segFtVLAD1,
    segFtVLAD2,
    gt,
    segRange2,
    imInds1,
    map_calculate,
    domain,
    source_paths,
    target_paths,
    results_csv,
    metrics_csv,
    evaluation_top_k=5,
    save_results=True,
):

    # RECALL CALCULATION
    normalize_vectors = bool(experiment_config["pca"])
    database_vectors = prepare_vectors(segFtVLAD1, normalize_vectors)
    query_vectors = prepare_vectors(segFtVLAD2, normalize_vectors)
    index = build_flat_l2_index(database_vectors)
    sims, matches = index.search(query_vectors, min(200, index.ntotal))
    # matches = matches.T[0]
    if save_results:
        out_folder = f"{workdir}/results/global/"
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        if not os.path.exists(f"{out_folder}/{experiment_name}"):
            os.makedirs(f"{out_folder}/{experiment_name}")
        
        pkl_file_results = f"{out_folder}/{experiment_name}/{dataset_name}_matches_sims_domain_{domain}__{experiment_config['results_pkl_suffix']}"
        data_to_save = {'sims': sims, 'matches': matches}
        # Saving the data to a pickle file
        with open(pkl_file_results, 'wb') as file:
            pickle.dump(data_to_save, file)
        print(f"Results saved to {pkl_file_results}")
    
    # For  now just take 50 of those 100 matches
    sims_50 = sims[:, :50]
    matches_50 = matches[:, :50]

    sims_50 =2-sims_50#.T[0]
    # matches_justfirstone = matches.T[0]
    # sims =2-sims.T[0]
    ranking_gt = gt
    if ranking_gt is None:
        ranking_gt = [np.asarray([], dtype=np.int64) for _ in target_paths]
    max_seg_preds = func_vpr.get_matches(
        matches_50,
        ranking_gt,
        sims_50,
        segRange2,
        imInds1,
        n=evaluation_top_k,
        method="max_seg_topk_wt_borda_Im",
    )

    csv_paths = save_retrieval_csv(
        max_seg_preds,
        source_paths,
        target_paths,
        results_csv,
        metrics_csv,
        ground_truth=gt,
        top_k=evaluation_top_k,
    )
    print("Ranked source/target results saved to", csv_paths["results"])

    max_seg_recalls = None
    if gt is not None:
        max_seg_recalls = func_vpr.calc_recall(max_seg_preds, gt, evaluation_top_k)
        print("Precision, recall, accuracy, and F1 saved to", csv_paths["metrics"])

    print("VLAD + PCA Results \n ")
    if map_calculate and gt is not None:
        # mAP calculation
        queries_results = func_vpr.convert_to_queries_results_for_map(max_seg_preds, gt)
        map_value = func_vpr.calculate_map(queries_results)
        print(f"Mean Average Precision (mAP): {map_value}")

    if max_seg_recalls is not None:
        print("Max Seg Logs: ", max_seg_recalls)
    else:
        print("No ground truth supplied; result CSV written without accuracy metrics.")
    
    return max_seg_recalls

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Global Place Recognition on Any Dataset. See place_rec_global_config.py to see how to give arguments.')
    parser.add_argument('--dataset', required=True, help='Dataset name') 
    parser.add_argument('--experiment', required=True, help='Experiment name') 
    parser.add_argument('--vocab-vlad',required=True, choices=['domain', 'map'], help='Vocabulary choice for VLAD. Options: map, domain.')
    parser.add_argument('--save-results', action='store_true', help='Save results to file')
    parser.add_argument('--save-vector-db', action='store_true', help='Save a reusable FAISS index, source metadata, and query vectors')
    parser.add_argument('--benchmark-vector-db', action='store_true', help='Benchmark FAISS lookup latency per query image')
    parser.add_argument('--vector-db-dir', default=None, help='Vector database output directory; defaults to <dataset>/out/vector_db')
    parser.add_argument('--benchmark-top-k', type=int, default=200, help='Number of nearest segments used by the lookup benchmark')
    parser.add_argument('--benchmark-warmup', type=int, default=1, help='Unmeasured benchmark passes')
    parser.add_argument('--benchmark-repeats', type=int, default=5, help='Measured benchmark passes')
    parser.add_argument('--ground-truth-csv', default=None, help='CSV containing correct source,target image pairs')
    parser.add_argument('--result-csv', default=None, help='Ranked source/target result CSV path')
    parser.add_argument('--metrics-csv', default=None, help='Precision/recall/accuracy/F1 CSV path')
    parser.add_argument('--evaluation-top-k', type=int, default=5, help='Image ranks written to result and metrics CSV files')


    topk_value = 5 # This gives all results from recall@1 to 5 
    map_calculate = False  #Mean average precision: False always except to replicate certain results in supplementary.

    args = parser.parse_args()
    if args.evaluation_top_k <= 0:
        raise ValueError("--evaluation-top-k must be positive")

    save_results = args.save_results 
    print("Save results: ", save_results)

    print(f"Vocabulary choice for VLAD (domain/map) is {args.vocab_vlad}")

    experiment_name = f"{args.experiment}_{args.dataset}_{current_time}"

    # Load dataset and experiment configurations
    dataset_config = datasets.get(args.dataset, {})
    if not dataset_config:
        raise ValueError(f"Dataset '{args.dataset}' not found in configuration.")

    experiment_config = experiments.get(args.experiment, {})
    if not experiment_config:
        raise ValueError(f"Experiment '{args.experiment}' not found in configuration.")
    if (
        args.save_vector_db or args.benchmark_vector_db
    ) and experiment_config["global_method_name"] != "SegLoc":
        raise ValueError("Vector database export currently supports SegLoc experiments only")

    print("The selected dataset config is: \n", dataset_config)
    print("The selected experiment config is: \n", experiment_config)

    cfg = dataset_config['cfg']


    _, workdir, dataPath1_r, dataPath2_q = resolve_dataset_paths(
        args.dataset, dataset_config
    )
    os.makedirs(workdir, exist_ok=True)

    vector_db_dir = args.vector_db_dir or os.path.join(workdir, "vector_db")
    vector_db_name = f"{args.dataset}_{args.experiment}_{args.vocab_vlad}"
    vector_db_paths = artifact_paths(vector_db_dir, vector_db_name)
    results_csv_path = args.result_csv or vector_db_paths["results"]
    metrics_csv_path = args.metrics_csv or vector_db_paths["metrics"]

    save_path_results = f"{workdir}/results/"

    cache_dir = './cache'

    device = torch.device("cuda")
    # Dino_v2 properties (parameters)
    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    num_c: int = 32
    # domain: Literal["aerial", "indoor", "urban"] =  dataset_config['domain_vlad_cluster']
    # if argument vocab-vlad is domain, then use domain_vlad_cluster, else use map_vlad_cluster
    domain = dataset_config['domain_vlad_cluster'] if args.vocab_vlad == 'domain' else dataset_config['map_vlad_cluster']
    print(f"IMPORTANT: domain is {domain}")
    ext_specifier = f"dinov2_vitg14/l{desc_layer}_{desc_facet}_c{num_c}"
    c_centers_file = os.path.join(cache_dir, "vocabulary", ext_specifier,
                                domain, "c_centers.pt")
    assert os.path.isfile(c_centers_file), "Cluster centers not cached!"
    c_centers = torch.load(c_centers_file)
    assert c_centers.shape[0] == num_c, "Wrong number of clusters!"

    vlad = VLAD(num_c, desc_dim=None, 
            cache_dir=os.path.dirname(c_centers_file))
    # Fit (load) the cluster centers (this'll also load the desc_dim)
    vlad.fit(None)

    #Load Descriptors
    dino_r_path = f"{workdir}/{dataset_config['dino_h5_filename_r']}"
    dino_q_path = f"{workdir}/{dataset_config['dino_h5_filename_q']}"
    dino1_h5_r = h5py.File(dino_r_path, 'r')
    dino2_h5_q = h5py.File(dino_q_path, 'r')

    ims_sidx, ims_eidx, ims_step = 0, None, 1
    ims1_r = natsorted(list_image_names(dataPath1_r))
    ims1_r = ims1_r[ims_sidx:ims_eidx][::ims_step]
    ims2_q = natsorted(list_image_names(dataPath2_q))
    ims2_q = ims2_q[ims_sidx:ims_eidx][::ims_step]
    source_paths = [
        os.path.abspath(os.path.join(dataPath1_r, image_name)) for image_name in ims1_r
    ]
    target_paths = [
        os.path.abspath(os.path.join(dataPath2_q, image_name)) for image_name in ims2_q
    ]

    # dataset specific ground truth
    gt = get_gt(
    dataset=args.dataset,
    cfg=cfg,
    workdir_data=workdir_data,
    ims1_r=ims1_r,          
    ims2_q=ims2_q,          
    func_vpr_module=func_vpr
    )
    if args.ground_truth_csv:
        gt = load_ground_truth_csv(
            args.ground_truth_csv, source_paths, target_paths
        )
        print("Ground truth loaded from", args.ground_truth_csv)

    if experiment_config["global_method_name"] == "SegLoc":
        dh = cfg['desired_height'] // 14
        dw = cfg['desired_width'] // 14
        idx_matrix = np.empty((cfg['desired_height'], cfg['desired_width'], 2)).astype('int32')
        for i in range(cfg['desired_height']):
            for j in range(cfg['desired_width']):
                idx_matrix[i, j] = np.array([np.clip(i//14, 0, dh-1), np.clip(j//14, 0, dw-1)])
        ind_matrix = np.ravel_multi_index(idx_matrix.reshape(-1, 2).T, (dh, dw))
        ind_matrix = torch.tensor(ind_matrix, device='cuda')


        masks_r_path = f"{workdir}/{dataset_config['masks_h5_filename_r']}"
        masks_q_path = f"{workdir}/{dataset_config['masks_h5_filename_q']}"
        masks1_h5_r = h5py.File(masks_r_path, 'r')
        masks2_h5_q = h5py.File(masks_q_path, 'r')

        order = experiment_config['order']
        print("nbr agg order number: ", order)

        segRange1 = []
        segRange2 = []
        desc_dim = 1536
        vlad_dim = 32 * desc_dim

        # For PCA
        total_segments = 0# Counter for sampled segments
        max_segments = 50000  # Max segments to sample in total

        batch_size = 100  # Number of images to process before applying PCA

        if experiment_config["pca"]:

            if args.vocab_vlad == 'domain': 
                pca_model_path = f"{workdir}/{args.dataset}{experiment_config['pca_model_pkl']}"
            elif args.vocab_vlad == 'map':
                pca_model_path = f"{workdir}/{args.dataset}{experiment_config['pca_model_pkl_map']}"
            else:
                raise ValueError(f"Unknown vocab-vlad value: {args.vocab_vlad}")
            # pca_model_path = f"{workdir}/{args.dataset}{experiment_config['pca_model_pkl']}"
        else: 
            pca_model_path = None



        segFtVLAD1_list = [] 
        segFtVLAD1Pca_list = [] 
        batch_descriptors_r = [] 

        segFtVLAD2_list = [] 
        segFtVLAD2Pca_list = [] 
        batch_descriptors_q = [] 

        imInds1 = np.array([], dtype=int)
        imInds2 = np.array([], dtype=int)

        execution_times_total = []

        print("Computing SegLoc for all images in the dataset...")
        for r_id, r_img in tqdm(enumerate(ims1_r), total=len(ims1_r), desc="Processing for reference images..."):

            # print(r_id, r_img)
            # Preload all masks for the image
            masks_seg = func_vpr.preload_masks(masks1_h5_r, r_img)

            imInds1_ind, regInds1_ind, segMask1_ind = func_vpr.getIdxSingleFast(r_id,masks_seg,minArea=experiment_config['minArea'])

            imInds1 = np.concatenate((imInds1, imInds1_ind))

            if order: 
                adjMat1_ind = func_vpr.nbrMasksAGGFastSingle(masks_seg, order)
            else:
                adjMat1_ind = None

            gd = func_vpr.seg_vlad_gpu_single(ind_matrix, idx_matrix, dino1_h5_r, r_img, segMask1_ind, c_centers, cfg, desc_dim=1536, adj_mat=adjMat1_ind)

            if experiment_config["pca"]:
                batch_descriptors_r.append(gd)
                if (r_id + 1) % batch_size == 0 or (r_id + 1) == len(ims1_r): ## Once we have accumulated descriptors for 100 images or are at the last image, process the batch
                    segFtVLAD1_batch = torch.cat(batch_descriptors_r, dim=0)
                    # Reset batch descriptors for the next batch
                    batch_descriptors_r = []

                    print("Applying PCA to batch descriptors... at image ", r_id)
                    segFtVLAD1Pca_batch  = func_vpr.apply_pca_transform_from_pkl(segFtVLAD1_batch, pca_model_path)
                    del segFtVLAD1_batch

                    segFtVLAD1Pca_list.append(segFtVLAD1Pca_batch)
                    

            else:
                segFtVLAD1_list.append(gd) #imfts_batch same as gd here, in the full image function, it is for 100 images at a time



        if experiment_config["pca"]:
            segFtVLAD1 = torch.cat(segFtVLAD1Pca_list, dim=0)
            print("Shape of segment descriptors with PCA:", segFtVLAD1.shape)
            del segFtVLAD1Pca_list
        else:
            segFtVLAD1 = torch.cat(segFtVLAD1_list, dim=0)
            print("Shape of segment descriptors without PCA:", segFtVLAD1.shape)
            del segFtVLAD1_list

        for i in range(imInds1[-1]+1):
            segRange1.append(np.where(imInds1==i)[0])

        vector_db_index = None
        vector_db_normalized = bool(experiment_config["pca"])
        if args.save_vector_db or args.benchmark_vector_db:
            database_vectors = prepare_vectors(segFtVLAD1, vector_db_normalized)
            vector_db_index = build_flat_l2_index(database_vectors)

            if args.save_vector_db:
                save_database(
                    vector_db_index,
                    database_vectors,
                    imInds1,
                    source_paths,
                    vector_db_dir,
                    vector_db_name,
                    vector_db_normalized,
                    extra_metadata={
                        "dataset": args.dataset,
                        "experiment": args.experiment,
                        "vocab_vlad": args.vocab_vlad,
                        "vlad_cluster": domain,
                        "pca": bool(experiment_config["pca"]),
                        "source_root": os.path.abspath(dataPath1_r),
                    },
                )
                print(f"FAISS vector database saved to {vector_db_paths['index']}")
                print(f"Source metadata saved to {vector_db_paths['metadata']}")

        if save_results:
            out_folder = f"{workdir}/results/global/"
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            if not os.path.exists(f"{out_folder}/{experiment_name}"):
                os.makedirs(f"{out_folder}/{experiment_name}")
            
            pkl_file_results1 = f"{out_folder}/{experiment_name}/{args.dataset}_segFtVLAD1_domain_{domain}__{experiment_config['results_pkl_suffix']}"

            # Saving segFtVLAD1
            with open(pkl_file_results1, 'wb') as file:
                pickle.dump(segFtVLAD1, file)
            print(f"segFtVLAD1 tensor saved to {pkl_file_results1}")


        # QUERIES
        for q_id, q_img in tqdm(enumerate(ims2_q), total=len(ims2_q), desc="Processing for query images..."):
            # print("query: ", q_id, q_img)
            # Preload all masks for the image
            masks_seg = func_vpr.preload_masks(masks2_h5_q, q_img)

            imInds2_ind, regInds2_ind, segMask2_ind = func_vpr.getIdxSingleFast(q_id,masks_seg,minArea=experiment_config['minArea'])

            imInds2 = np.concatenate((imInds2, imInds2_ind))

            if order: 
                adjMat2_ind = func_vpr.nbrMasksAGGFastSingle(masks_seg, order)
            else:
                adjMat2_ind = None

            gd = func_vpr.seg_vlad_gpu_single(ind_matrix, idx_matrix, dino2_h5_q, q_img, segMask2_ind, c_centers, cfg, desc_dim=1536, adj_mat=adjMat2_ind)


            if experiment_config["pca"]:
                batch_descriptors_q.append(gd)
                if (q_id + 1) % batch_size == 0 or (q_id + 1) == len(ims2_q): ## Once we have accumulated descriptors for 100 images or are at the last image, process the batch
                    segFtVLAD2_batch = torch.cat(batch_descriptors_q, dim=0)
                    # Reset batch descriptors for the next batch
                    batch_descriptors_q = []

                    print("query: Applying PCA to batch descriptors... at image ", q_id)
                    segFtVLAD2Pca_batch  = func_vpr.apply_pca_transform_from_pkl(segFtVLAD2_batch, pca_model_path)
                    del segFtVLAD2_batch

                    segFtVLAD2Pca_list.append(segFtVLAD2Pca_batch)
                    

            else:
                segFtVLAD2_list.append(gd) #imfts_batch same as gd here, in the full image function, it is for 100 images at a time



        if experiment_config["pca"]:
            segFtVLAD2 = torch.cat(segFtVLAD2Pca_list, dim=0)
            print("Shape of q segment descriptors with PCA:", segFtVLAD2.shape)
            del segFtVLAD2Pca_list
        else:
            segFtVLAD2 = torch.cat(segFtVLAD2_list, dim=0)
            print("Shape of q segment descriptors without PCA:", segFtVLAD2.shape)
            del segFtVLAD2_list

        for i in range(imInds2[-1]+1):
            segRange2.append(np.where(imInds2==i)[0])

        if args.save_vector_db or args.benchmark_vector_db:
            query_vectors = prepare_vectors(segFtVLAD2, vector_db_normalized)
            query_offsets = build_image_offsets(imInds2, len(ims2_q))

            if args.save_vector_db:
                save_queries(
                    query_vectors,
                    imInds2,
                    target_paths,
                    vector_db_paths["queries"],
                    vector_db_normalized,
                )
                print(f"Query benchmark vectors saved to {vector_db_paths['queries']}")

            if args.benchmark_vector_db:
                benchmark_target = vector_db_index
                if args.save_vector_db:
                    # Reload the serialized index so the benchmark validates the artifact.
                    benchmark_target = faiss.read_index(vector_db_paths["index"])
                benchmark_result = benchmark_index(
                    benchmark_target,
                    query_vectors,
                    query_offsets,
                    top_k=args.benchmark_top_k,
                    warmup=args.benchmark_warmup,
                    repeats=args.benchmark_repeats,
                )
                benchmark_result.update({
                    "dataset": args.dataset,
                    "experiment": args.experiment,
                    "vocab_vlad": args.vocab_vlad,
                    "faiss_threads": faiss.omp_get_max_threads(),
                })
                save_benchmark(benchmark_result, vector_db_paths["benchmark"])
                print("FAISS per-image lookup benchmark:")
                print(json.dumps(benchmark_result, indent=2, sort_keys=True))
                print(f"Benchmark saved to {vector_db_paths['benchmark']}")

        if save_results:
            out_folder = f"{workdir}/results/global/"
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            if not os.path.exists(f"{out_folder}/{experiment_name}"):
                os.makedirs(f"{out_folder}/{experiment_name}")
            
            pkl_file_results2 = f"{out_folder}/{experiment_name}/{args.dataset}_segFtVLAD2_domain_{domain}__{experiment_config['results_pkl_suffix']}"

            # Saving segFtVLAD2
            with open(pkl_file_results2, 'wb') as file:
                pickle.dump(segFtVLAD2, file)
            print(f"segFtVLAD2 tensor saved to {pkl_file_results2}")

        # RETRIEVAL RESULTS AND METRICS
        recall_segrec = recall_segloc(
            workdir,
            args.dataset,
            experiment_config,
            experiment_name,
            segFtVLAD1,
            segFtVLAD2,
            gt,
            segRange2,
            imInds1,
            map_calculate,
            domain,
            source_paths,
            target_paths,
            results_csv_path,
            metrics_csv_path,
            args.evaluation_top_k,
            save_results,
        )
        if gt is not None:
            print("VLAD + PCA RESULTS for segloc for dataset config: ", dataset_config, " ::: experiment config ::: ", experiment_config, " using pca file : ", pca_model_path, "experiment_name: ", experiment_name)
            print(recall_segrec)



    elif experiment_config["global_method_name"] == "AnyLoc":
        print("Running AnyLoc for global place recognition...")

        # VLAD
        imFts1_vlad = func_vpr.aggFt(dino_r_path,None,None,cfg,'vlad',vlad,upsample=True)
        imFts2_vlad = func_vpr.aggFt(dino_q_path,None,None,cfg,'vlad',vlad,upsample=True)

        topk_value = topk_value #10 #5

        print ("We use implementation of AnyLoc that does NOT use PCA... i.e. AnyLoc-VLAD-DINOv2 in their paper")
        recall_VLAD_pca, match_info = func_vpr.get_recall(func_vpr.normalizeFeat(imFts1_vlad), func_vpr.normalizeFeat(imFts2_vlad),gt, k=topk_value)
        print("RESULTS for anyloc: VLAD PCA:  ")
        print(recall_VLAD_pca)

        if save_results:
            print("saving anyloc results...")
            out_folder = f"{workdir}/results/global/"
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            if not os.path.exists(f"{out_folder}/anyloc"):
                os.makedirs(f"{out_folder}/anyloc")
            
            pkl_file_results1 = f"{out_folder}/anyloc/{args.dataset}_segFtVLAD1{experiment_config['results_pkl_suffix']}"
            pkl_file_results2 = f"{out_folder}/anyloc/{args.dataset}_segFtVLAD2{experiment_config['results_pkl_suffix']}"

            with open(pkl_file_results1, 'wb') as file:
                pickle.dump(imFts1_vlad, file)
            with open(pkl_file_results2, 'wb') as file:
                pickle.dump(imFts2_vlad, file)
            print(f"ANYLOC SAVING DONE: segFtVLAD1 and 2 tensor saved to {pkl_file_results1} and {pkl_file_results2}")


            # print("VLAD + PCA Results \n ")
            # if map_calculate:
            #     formatted_max_seg_preds = [item['img_id_r'].tolist() for item in match_info]
            #     # mAP calculation
            #     queries_results = func_vpr.convert_to_queries_results_for_map(formatted_max_seg_preds, gt)
            #     map_value = func_vpr.calculate_map(queries_results)
            #     print(f"Mean Average Precision (mAP) for VLAD + PCA: {map_value} for top {topk_value} matches.")


    else:
        raise ValueError(f"Global Method '{experiment_config['global_method_name']}' not found in configuration.")

    print("Script fully Executed! Check your results!")
