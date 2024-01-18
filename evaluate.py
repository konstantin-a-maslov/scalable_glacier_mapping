import utils
import h5py
import numpy as np
import pickle
import config
import os
from tqdm import tqdm

  
def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def precision(tp, tn, fp, fn):
    return tp / (tp + fp)


def recall(tp, tn, fp, fn):
    return tp / (tp + fn)


def f1(tp, tn, fp, fn):
    precision_ = precision(tp, tn, fp, fn)
    recall_ = recall(tp, tn, fp, fn)
    return 2 * precision_ * recall_ / (precision_ + recall_)


def iou(tp, tn, fp, fn):
    return tp / (tp + fp + fn)


def populate_evaluation_dict(tp, tn, fp, fn):
    evaluation_dict = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    evaluation_dict["accuracy"] = accuracy(tp, tn, fp, fn)
    evaluation_dict["precision"] = precision(tp, tn, fp, fn)
    evaluation_dict["recall"] = recall(tp, tn, fp, fn)
    evaluation_dict["f1"] = f1(tp, tn, fp, fn)
    evaluation_dict["iou"] = iou(tp, tn, fp, fn)
    return evaluation_dict


def report(evaluation):
    ious = []
    print()
    regions = sorted(evaluation["regions"].keys())
    for region in regions:
        iou = evaluation["regions"][region]["iou"]
        ious.append(iou)
        print(f"{region}: \t {iou}")
    print()
    print(f"Average: \t {np.mean(ious)}")
    print(f"Std.dev.: \t {np.std(ious)}")
    print()
    subregions = sorted(evaluation["subregions"].keys())
    for subregion in subregions:
        iou = evaluation["subregions"][subregion]["iou"]
        print(f"{subregion}: \t {iou}")
    print()


def main():
    utils.update_config_from_cli(config)

    predictions_dataset_path = os.path.join(config.data.predictions_dir, config.model.model_name, "predictions.hdf5")
    predictions_dataset = h5py.File(predictions_dataset_path, "r")

    evaluation = {}
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

    regions = set()
    subregions = set()

    for tile_name in tqdm(predictions_dataset.keys()):
        tile = predictions_dataset[tile_name]
        true = np.array(tile["true"])
        pred = np.array(tile["pred"])

        tp = np.sum((pred == 1) & (true == 1))
        tn = np.sum((pred == 0) & (true == 0))
        fp = np.sum((pred == 1) & (true == 0))
        fn = np.sum((pred == 0) & (true == 1))
        evaluation[tile_name] = populate_evaluation_dict(tp, tn, fp, fn)

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        region = tile.attrs["region"]
        regions.add(region)
        subregion = tile.attrs["subregion"]
        subregions.add(subregion)

    evaluation["total"] = populate_evaluation_dict(total_tp, total_tn, total_fp, total_fn)

    evaluation["regions"] = {}
    for region in tqdm(regions):
        region_tp, region_tn, region_fp, region_fn = 0, 0, 0, 0
        for tile_name in predictions_dataset.keys():
            tile = predictions_dataset[tile_name]
            tile_region = tile.attrs["region"]
            if not region == tile_region:
                continue
            region_tp += evaluation[tile_name]["tp"]
            region_tn += evaluation[tile_name]["tn"]
            region_fp += evaluation[tile_name]["fp"]
            region_fn += evaluation[tile_name]["fn"]
        evaluation["regions"][region] = populate_evaluation_dict(region_tp, region_tn, region_fp, region_fn)

    evaluation["subregions"] = {}
    for subregion in tqdm(subregions):
        subregion_tp, subregion_tn, subregion_fp, subregion_fn = 0, 0, 0, 0
        for tile_name in predictions_dataset.keys():
            tile = predictions_dataset[tile_name]
            tile_subregion = tile.attrs["subregion"]
            if not subregion == tile_subregion:
                continue
            subregion_tp += evaluation[tile_name]["tp"]
            subregion_tn += evaluation[tile_name]["tn"]
            subregion_fp += evaluation[tile_name]["fp"]
            subregion_fn += evaluation[tile_name]["fn"]
        evaluation["subregions"][subregion] = populate_evaluation_dict(subregion_tp, subregion_tn, subregion_fp, subregion_fn)

    with open(os.path.join(config.data.predictions_dir, config.model.model_name, "evaluation.pickle"), "wb") as evaluation_output:
        pickle.dump(evaluation, evaluation_output, protocol=pickle.HIGHEST_PROTOCOL)

    report(evaluation)       
    predictions_dataset.close() 
            

if __name__ == "__main__":
    main()
    