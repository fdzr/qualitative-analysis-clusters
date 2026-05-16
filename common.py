import json
import sys
import logging
import typing
import unicodedata
from pathlib import Path
import copy
import random
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
from scipy.stats import spearmanr

from custom_types import ShortUse, Results

random.seed(42)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def get_adj_matrix(
    scores: pd.DataFrame,
    id2int: dict,
    n_sentences,
    fill_diagonal: bool,
    normalize: bool,
    threshold: float | None = None,
):
    logging.info("building adjacency matrix ...")
    matrix = np.zeros((n_sentences, n_sentences), dtype="float")

    if fill_diagonal is True:
        if normalize is True:
            np.fill_diagonal(matrix, 1.0)
        else:
            np.fill_diagonal(matrix, 4.0)

    if normalize is True:
        scaler = MinMaxScaler()
        scores["score"] = scaler.fit_transform(
            scores["score"].to_numpy().reshape(-1, 1)
        )

    for _, row in scores.iterrows():
        x = id2int[ShortUse(row["word"], row["identifier1"])]
        y = id2int[ShortUse(row["word"], row["identifier2"])]

        try:
            score = float(row["prediction"])
        except Exception as e:
            print(scores.word)
            sys.exit(1)

        if normalize is True:
            if threshold is not None and score < threshold:
                continue

        matrix[x, y] = matrix[y, x] = score

    logging.info("adjacency matrix built ...")

    return matrix


def compute_jsd(predictions: dict, grouping: pd.DataFrame, method: str = None):
    clusters_to_freq1 = {}
    clusters_to_freq2 = {}

    old_ids_samples = set(grouping[grouping["grouping"] == 1]["ids"].to_list())
    new_ids_samples = set(grouping[grouping["grouping"] == 2]["ids"].to_list())

    if method is None or method != "spectral_clustering":
        for id, cluster in predictions.items():
            if cluster not in clusters_to_freq1:
                clusters_to_freq1[cluster] = 0
            if cluster not in clusters_to_freq2:
                clusters_to_freq2[cluster] = 0

            if id in old_ids_samples:
                clusters_to_freq1[cluster] += 1
            if id in new_ids_samples:
                clusters_to_freq2[cluster] += 1

        c1 = np.array(list(clusters_to_freq1.values()))
        c2 = np.array(list(clusters_to_freq2.values()))
        val = distance.jensenshannon(c1, c2, base=2.0)
        answer = Results(
            jsd=val,
            cluster_to_freq1=clusters_to_freq1,
            cluster_to_freq2=clusters_to_freq2,
        )

    return answer


def load_data(path: str, wic_data=False):
    logging.info(f"loading data from {path} ...")

    if wic_data is True:
        data = pd.read_csv(f"{path}.scores")
        return data

    data = pd.read_csv(path)

    mask = data["prediction"] == "-"
    filtered_data = data[~mask]

    logging.info("data loaded ...")

    return filtered_data


def get_gold_data(p: str = "../test_data_es.csv"):
    logging.info("  loading gold data ...")
    gold_data = pd.read_csv(p, sep="\t")
    try:
        result = dict(
            zip(
                gold_data["lemma"],
                zip(
                    gold_data["change_graded"],
                    gold_data["change_binary"],
                ),
            )
        )
    except Exception as e:
        result = dict(
            zip(
                gold_data["word"],
                zip(
                    gold_data["change_graded"],
                    gold_data["change_binary"],
                ),
            )
        )

    logging.info("  gold data loaded ...")

    return result


def save_results(word: str, result: Results, parameters: dict, path_to_save: str):
    df = pd.read_csv(path_to_save)

    n_rows = df.shape[0]

    df.loc[n_rows, "word"] = word
    df.loc[n_rows, "jsd"] = result.jsd
    df.loc[n_rows, "clusters_to_freq1"] = str(result.cluster_to_freq1)
    df.loc[n_rows, "clusters_to_freq2"] = str(result.cluster_to_freq2)
    df.loc[n_rows, "parameters"] = str(parameters)

    df.to_csv(path_to_save, index=False)


def save_correlation(correlation: float, parameters: dict, path_to_file: str):
    df = pd.read_csv(path_to_file)

    n_rows = df.shape[0]

    df.loc[n_rows, "correlation"] = str(correlation)
    df.loc[n_rows, "parameters"] = str(parameters)

    df.to_csv(path_to_file, index=False)


def calculate_correlation(jsd: dict[str, Results], path_to_gold_data):
    gold_data = get_gold_data(path_to_gold_data)
    pred_change_graded = []
    gold_change_graded = []

    for word in gold_data:
        proccesed_word = unicodedata.normalize("NFC", word)

        try:
            pred_change_graded.append(jsd[proccesed_word].jsd)
            gold_change_graded.append(gold_data[proccesed_word][0])
        except Exception as e:
            logging.warning(f"    {word} is not a tw from the competence ...")

    spr = spearmanr(gold_change_graded, pred_change_graded)[0]
    return spr


def get_scaler(scores: pd.Series):
    return MinMaxScaler().fit(np.array(scores).reshape(-1, 1))


def get_thresholds(scores: pd.Series):
    return [0.5] + list(np.quantile(scores, np.arange(0.1, 1.0, 0.1)))


def _next_run_id(experiments_path: str):
    runs = sorted(Path(experiments_path, "runs").glob("run_*"))
    return f"run_{len(runs) + 1:03d}"


def save_cluster_assignments(
    pred_clusters: dict,
    word: str,
    run_path: Path,
    sentences: dict,
):
    grouped = defaultdict(list)
    for sentence_id, cluster_label in pred_clusters.items():
        grouped[int(cluster_label)].append(
            {
                "id": sentence_id,
                "text": sentences.get(sentence_id, ""),
            }
        )

    data = {
        "word": word,
        "clusters": [
            {"id": cid, "sentences": sents}
            for cid, sents in sorted(
                grouped.items(),
            )
        ],
    }

    clusters_dir = run_path / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)
    (clusters_dir / f"{word}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False)
    )


def save_run_metadata(params: dict, spearman: float, run_path: Path):
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "params.json").write_text(json.dumps(params, indent=2))
    (run_path / "spearman.json").write_text(
        json.dumps({"spearman": spearman}, indent=2)
    )


def update_summary(run_id: str, spearman: float, params: dict, experiments_path: Path):
    summary_path = Path(experiments_path, "summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    summary.append({"run": run_id, "spearman": spearman, "params": params})
    summary.sort(key=lambda x: x["spearman"], reverse=True)
    summary_path.write_text(json.dumps(summary, indent=2))


def generate_hyperparameter_combinations(
    model_hyperparameter_combinations: list, fill_diagonal: bool, normalize: bool
):
    hyperparameter_combinations = []

    for fd in [True]:
        for nm in [False] if normalize is False else [False, True]:
            for combination in model_hyperparameter_combinations:
                if "distribution" in combination:
                    if (
                        combination["distribution"].startswith("discrete")
                        and nm is True
                    ):
                        continue
                    if combination["distribution"].startswith("real") and nm is False:
                        continue

                hyperparameter_combinations.append(
                    {
                        "fill_diagonal": fd,
                        "normalize": nm,
                        "model_hyperparameters": combination,
                    }
                )

    return hyperparameter_combinations


def get_predictions(
    get_clusters: typing.Callable,
    scores: pd.DataFrame,
    hyperparameter_combinations: typing.List[dict],
    metadata: dict,
    run_path: Path,
):
    logging.info("get predictions ...")
    words = scores.word.unique()
    jsd = {}

    method = metadata["method"]
    dataset = metadata["dataset"]
    name_file = metadata["name_file"]

    for word in words:
        mask = scores["word"] == word
        filtered_scores = scores[mask]

        ids = set(filtered_scores["identifier1"].to_list()).union(
            set(filtered_scores["identifier2"].to_list())
        )

        grouping = pd.DataFrame({"ids": list(ids)})
        grouping["grouping"] = grouping.apply(
            lambda row: 1 if row["ids"].startswith("old") else 2, axis=1
        )

        context = [ShortUse(word=word, id=id) for id in ids]
        n_sentences = len(ids)

        id2int = {value: index for index, value in enumerate(context)}

        adj_matrix = get_adj_matrix(
            filtered_scores,
            id2int,
            n_sentences,
            hyperparameter_combinations["fill_diagonal"],
            hyperparameter_combinations["normalize"],
        )

        distance_matrix = adj_matrix.max() - adj_matrix
        best_silhouette = -1
        best_labels = None

        for n in range(2, 6):
            hyperparams = {
                **hyperparameter_combinations["model_hyperparameters"],
                "n_clusters": n,
            }

            labels = get_clusters(
                distance_matrix,
                hyperparameter_combinations["model_hyperparameters"],
            )
            score = silhouette_score(
                distance_matrix,
                labels,
                metric="precomputed",
            )
            logging.info(f" n_clusters={n} silhouette={score:.4f}")
            if score > best_silhouette:
                best_silhouette = score
                best_labels = labels

        logging.info(
            f" best n_clusters={best_labels.max() + 1} silhouette={best_silhouette:.4f}"
        )

        pred_clusters = {
            c.id: best_labels[id2int[c]] for index, c in enumerate(context)
        }

        id1_map = (
            filtered_scores[["identifier1", "sentence1"]]
            .drop_duplicates("identifier1")
            .set_index("identifier1")["sentence1"]
            .to_dict()
        )

        id2_map = (
            filtered_scores[["identifier2", "sentence2"]]
            .drop_duplicates("identifier2")
            .set_index("identifier2")["sentence2"]
            .to_dict()
        )

        sentences = {**id1_map, **id2_map}

        save_cluster_assignments(
            pred_clusters,
            word,
            run_path,
            sentences,
        )
        # save_cluster_assignments(
        #     pred_clusters,
        #     experiment_id,
        #     word,
        #     path,
        # )
        jsd[word] = compute_jsd(pred_clusters, grouping)
        # save_results(
        #     word,
        #     jsd[word],
        #     hyperparameter_combinations,
        #     f"./results/{method}/{dataset}/full_experiment.csv",
        # )

    logging.info("returning predictions ...")

    return jsd


def eval(
    get_clusters: typing.Callable,
    scores: pd.DataFrame,
    parameters: dict,
    metadata: dict,
):
    logging.info(f"eval {metadata['method']} method...")

    metadata["name_file"] = "results_testing_set"

    experiments_path = f"./results/{metadata['method']}/{metadata['dataset']}"
    Path(experiments_path, "runs").mkdir(parents=True, exist_ok=True)

    results = {}

    for hyperparameters in parameters:
        if metadata["method"] in ["ac", "sc"]:
            run_id = _next_run_id(experiments_path)
            run_path = Path(experiments_path, "runs", run_id)

            jsd = get_predictions(
                get_clusters,
                scores,
                hyperparameters,
                metadata=metadata,
                run_path=run_path,
            )
        # else:
        #     jsd = get_predictions_without_nclusters(
        #         get_clusters, scores, hyperparameters, metadata=metadata
        #     )

        logging.info("  calculating correlation ...")
        spr = calculate_correlation(jsd, metadata["path_to_gold_data"])
        logging.info(" correlation calculated ...")

        logging.info("  saving results ...")
        save_run_metadata(
            hyperparameters,
            spr,
            run_path,
        )
        update_summary(
            run_id,
            spr,
            hyperparameters,
            experiments_path,
        )
        logging.info("  results saved ...")

        results[run_id] = spr

    return results


def grid_search(
    get_data: typing.Callable,
    get_clusters: typing.Callable,
    model_hyperameter_combinations,
    metadata: dict = None,
):

    data = load_data(
        metadata["path_to_data"],
        wic_data=metadata["wic_data"],
    )

    hyperparameter_combinations = generate_hyperparameter_combinations(
        model_hyperameter_combinations,
        metadata["fill_diagonal"],
        metadata["normalize"],
    )

    eval(
        get_clusters,
        data,
        hyperparameter_combinations,
        metadata,
    )
