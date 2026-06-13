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
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import KFold
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
        scores["prediction"] = scaler.fit_transform(
            scores["prediction"].to_numpy().reshape(-1, 1)
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


def load_data(path: str):
    logging.info(f"loading data from {path} ...")

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


def calculate_correlation_for_words(
    jsd: dict,
    path_to_gold_data: str,
    words: list,
) -> float:

    gold_data = get_gold_data(path_to_gold_data)
    pred_change_graded = []
    gold_change_graded = []

    for word in words:
        processed_word = unicodedata.normalize("NFC", word)
        if processed_word not in jsd:
            continue
        if processed_word not in gold_data:
            logging.warning(f"  {word} not found in gold data ...")
            continue

        pred_change_graded.append(jsd[processed_word].jsd)
        gold_change_graded.append(gold_data[processed_word][0])

    if len(pred_change_graded) < 2:
        return 0.0

    return spearmanr(gold_change_graded, pred_change_graded)[0]


def get_scaler(scores: pd.Series):
    return MinMaxScaler().fit(np.array(scores).reshape(-1, 1))


def get_thresholds(scores: pd.Series):
    return [(i, float(np.quantile(scores, i / 10))) for i in range(1, 10)]


def _next_run_id(experiments_path: str):
    runs = sorted(Path(experiments_path, "runs").glob("run_*"))
    return f"run_{len(runs) + 1:03d}"


def generate_and_save_folds(
    scores: pd.DataFrame,
    experiments_path: str,
    k: int = 5,
    seed: int = 42,
) -> dict:

    words = sorted(scores.word.unique())
    kf = KFold(
        n_splits=k,
        shuffle=True,
        random_state=seed,
    )

    folds = {}
    for idx, (train_idx, test_idx) in enumerate(kf.split(words), start=1):
        folds[f"fold_{idx}"] = {
            "train": [words[j] for j in train_idx],
            "test": [words[j] for j in test_idx],
        }

    cv_path = Path(experiments_path, "cv")
    cv_path.mkdir(parents=True, exist_ok=True)
    (cv_path / "folds.json").write_text(json.dumps(folds, indent=2))

    return folds


def load_gold_senses(path: str):
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["identifier"], df["cluster"]))


def build_english_gold_index(gold_dir: str) -> dict:
    index = {}
    for path in Path(gold_dir).glob("*.csv"):
        word = path.stem.rsplit("_", 1)[0]
        index[word] = str(path)

    return index


def build_spanish_gold_index(gold_dir: str) -> dict:
    index = {}
    for path in Path(gold_dir).glob("*.csv"):
        word = unicodedata.normalize("NFC", path.stem)
        index[word] = str(path)

    return index


def get_spanish_gold_path(gold_dir: str, word: str):
    normalized_word = unicodedata.normalize("NFC", word)
    return str(Path(gold_dir) / f"{normalized_word}.csv")


def compute_ari_for_word(pred_clusters: dict, gold_senses: dict) -> float:
    common_ids = sorted(set(pred_clusters.keys()) & set(gold_senses.keys()))
    if len(common_ids) < 2:
        return 0.0

    pred_labels = [int(pred_clusters[id]) for id in common_ids]
    gold_labels = [int(gold_senses[id]) for id in common_ids]

    return adjusted_rand_score(gold_labels, pred_labels)


def calculate_ari_across_words(
    pred_clusters_per_word: dict,
    words: list,
    gold_dir: str,
    dataset: str,
) -> float:

    if dataset == "dwug_es":
        index = build_spanish_gold_index(gold_dir)
    else:
        index = build_english_gold_index(gold_dir)

    scores = []
    for word in words:
        if word not in pred_clusters_per_word:
            continue

        normalized_word = unicodedata.normalize("NFC", word)
        gold_path = index.get(normalized_word)
        if gold_path is None:
            logging.warning(f"No gold file found for word: {word}")
            continue

        gold_senses = load_gold_senses(gold_path)
        ari = compute_ari_for_word(pred_clusters_per_word[word], gold_senses)
        scores.append(ari)

    return sum(scores) / len(scores) if scores else 0.0


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
        "n_clusters": len(grouped),
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
    model_hyperparameter_combinations: list,
    normalize: bool,
    thresholds: typing.List | None,
):
    hyperparameter_combinations = []

    for fd in [True, False]:
        for nm in [False] if normalize is False else [False, True]:
            for quantile, threshold in (thresholds if thresholds else [(None, None)]):
                for combination in model_hyperparameter_combinations:
                    if "distribution" in combination:
                        if (
                            combination["distribution"].startswith("discrete")
                            and normalize is True
                        ):
                            continue
                        if (
                            combination["distribution"].startswith("real")
                            and normalize is False
                        ):
                            continue

                    hyperparameter_combinations.append(
                        {
                            "fill_diagonal": fd,
                            "normalize": nm,
                            "quantile": quantile,
                            "threshold": threshold,
                            "model_hyperparameters": combination,
                        }
                    )

    return hyperparameter_combinations


def cross_validate(
    get_clusters: typing.Callable,
    model_hyperparameter_combinations: list,
    metadata: dict,
    gold_dir: str,
    k: int = 5,
):
    data = load_data(metadata["path_to_data"])
    thresholds = (
        get_thresholds(data["prediction"])
        if metadata.get("use_threshold", False)
        else None
    )

    hyperparameter_combinations = generate_hyperparameter_combinations(
        model_hyperparameter_combinations,
        metadata["normalize"],
        thresholds,
    )

    experiments_path = (
        f"./results/{metadata['method']}/{metadata['dataset']}/{metadata['model']}"
    )
    folds = generate_and_save_folds(data, experiments_path, k=k)
    cv_results = []

    for fold_name, fold_data in folds.items():
        logging.info(f"processing {fold_name} ...")

        train_words = fold_data["train"]
        test_words = fold_data["test"]

        train_scores = data[data["word"].isin(train_words)]
        test_scores = data[data["word"].isin(test_words)]

        fold_path = Path(experiments_path, "cv", fold_name)
        fold_path.mkdir(parents=True, exist_ok=True)

        best_train_ari = -1
        best_params_ari = None
        best_train_lscd = -1
        best_params_lscd = None

        for hyperparameters in hyperparameter_combinations:
            dummy_path = fold_path / "train_temp"

            if metadata["method"] in ["sc", "ac"]:
                train_jsd, train_pred = get_predictions(
                    get_clusters,
                    train_scores,
                    hyperparameters,
                    metadata=metadata,
                    run_path=dummy_path,
                    save_clusters=False,
                )
            else:
                train_jsd, train_pred = get_predictions_no_clusters(
                    get_clusters,
                    train_scores,
                    hyperparameters,
                    metadata=metadata,
                    run_path=dummy_path,
                    save_clusters=False,
                )

            train_ari = calculate_ari_across_words(
                train_pred,
                train_words,
                gold_dir,
                metadata["dataset"],
            )
            train_lscd = calculate_correlation_for_words(
                train_jsd,
                metadata["path_to_gold_data"],
                train_words,
            )

            if train_ari > best_train_ari:
                best_train_ari = train_ari
                best_params_ari = hyperparameters

            if train_lscd > best_train_lscd:
                best_train_lscd = train_lscd
                best_params_lscd = hyperparameters

        test_path_ari = fold_path / "protocol_ari"
        if metadata["method"] in ["ac", "sc"]:
            test_jsd_ari, test_pred_ari = get_predictions(
                get_clusters,
                test_scores,
                best_params_ari,
                metadata=metadata,
                run_path=test_path_ari,
                save_clusters=True,
            )
        else:
            test_jsd_ari, test_pred_ari = get_predictions_no_clusters(
                get_clusters,
                test_scores,
                best_params_ari,
                metadata=metadata,
                run_path=test_path_ari,
                save_clusters=True,
            )

        test_ari_p1 = calculate_ari_across_words(
            test_pred_ari,
            test_words,
            gold_dir,
            metadata["dataset"],
        )
        test_lscd_p1 = calculate_correlation_for_words(
            test_jsd_ari,
            metadata["path_to_gold_data"],
            test_words,
        )

        test_path_lscd = fold_path / "protocol_lscd"
        if metadata["method"] in ["ac", "sc"]:
            test_jsd_lscd, test_pred_lscd = get_predictions(
                get_clusters,
                test_scores,
                best_params_lscd,
                metadata=metadata,
                run_path=test_path_lscd,
                save_clusters=True,
            )
        else:
            test_jsd_lscd, test_pred_lscd = get_predictions_no_clusters(
                get_clusters,
                test_scores,
                best_params_lscd,
                metadata=metadata,
                run_path=test_path_lscd,
                save_clusters=True,
            )

        test_lscd_p2 = calculate_correlation_for_words(
            test_jsd_lscd,
            metadata["path_to_gold_data"],
            test_words,
        )
        test_ari_p2 = calculate_ari_across_words(
            test_pred_lscd,
            test_words,
            gold_dir,
            metadata["dataset"],
        )

        fold_results = {
            "protocol_ari": {
                "best_params": best_params_ari,
                "train_ari": best_train_ari,
                "test_ari": test_ari_p1,
                "test_lscd": test_lscd_p1,
            },
            "protocol_lscd": {
                "best_params": best_params_lscd,
                "train_lscd": best_train_lscd,
                "test_lscd": test_lscd_p2,
                "test_ari": test_ari_p2,
            },
        }

        (fold_path / "results.json").write_text(json.dumps(fold_results, indent=2))
        cv_results.append(fold_results)

        logging.info(
            f"{fold_name} - P1: test_ari={test_ari_p1:.4f} test_lscd={test_lscd_p1:.4f} | "
            f"P2: test_lscd={test_lscd_p2:.4f} test_ari={test_ari_p2:.4f}"
        )

    cv_summary = {
        "protocol_ari": {
            "avg_test_ari": sum(r["protocol_ari"]["test_ari"] for r in cv_results) / k,
            "avg_test_lscd": sum(r["protocol_ari"]["test_lscd"] for r in cv_results)
            / k,
        },
        "protocol_lscd": {
            "avg_test_lscd": sum(r["protocol_lscd"]["test_lscd"] for r in cv_results)
            / k,
            "avg_test_ari": sum(r["protocol_lscd"]["test_ari"] for r in cv_results) / k,
        },
    }

    cv_path = Path(experiments_path, "cv")
    (cv_path / "cv_summary.json").write_text(json.dumps(cv_summary, indent=2))

    logging.info(
        f"CV done - P1: avg_test_ari={cv_summary['protocol_ari']['avg_test_ari']:.4f} | "
        f"P2: avg_test_lscd={cv_summary['protocol_lscd']['avg_test_lscd']:.4f}"
    )

    return cv_summary


def get_predictions(
    get_clusters: typing.Callable,
    scores: pd.DataFrame,
    hyperparameter_combinations: typing.List[dict],
    metadata: dict,
    run_path: Path,
    save_clusters: bool = True,
):
    logging.info("get predictions ...")
    words = scores.word.unique()
    jsd = {}
    pred_clusters_per_word = {}

    for word in words:
        mask = scores["word"] == word
        filtered_scores = scores[mask]

        ids = sorted(
            set(filtered_scores["identifier1"].to_list()).union(
                set(filtered_scores["identifier2"].to_list())
            )
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
            hyperparameter_combinations.get("threshold", None),
        )

        distance_matrix = adj_matrix.max() - adj_matrix
        np.fill_diagonal(distance_matrix, 0)
        best_silhouette = -1
        best_labels = None

        for n in range(2, 6):
            hyperparams = {
                **hyperparameter_combinations["model_hyperparameters"],
                "n_clusters": n,
            }

            labels = get_clusters(
                adj_matrix,
                hyperparams,
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

        pred_clusters_per_word[word] = pred_clusters

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

        if save_clusters:
            save_cluster_assignments(
                pred_clusters,
                word,
                run_path,
                sentences,
            )

        jsd[word] = compute_jsd(
            pred_clusters,
            grouping,
        )

    logging.info("returning predictions ...")

    return jsd, pred_clusters_per_word


def get_predictions_no_clusters(
    get_clusters: typing.Callable,
    scores: pd.DataFrame,
    hyperparameter_combinations: dict,
    metadata: dict,
    run_path: Path,
    save_clusters: bool = True,
):
    logging.info("get predictions ...")
    words = scores.word.unique()
    jsd = {}
    pred_clusters_per_word = {}

    for word in words:
        mask = scores["word"] == word
        filtered_scores = scores[mask]

        ids = sorted(
            set(filtered_scores["identifier1"].to_list()).union(
                set(filtered_scores["identifier2"].to_list())
            )
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
            hyperparameter_combinations.get("threshold", None),
        )

        logging.info(f"calculating clusters for word: {word} ...")
        best_labels = get_clusters(
            adj_matrix,
            hyperparameter_combinations["model_hyperparameters"],
        )
        if best_labels is None or len(best_labels) == 0:
            logging.warning(f"Skipping word {word} - clustering timed out")
            continue

        logging.info(f" n_clusters found={best_labels.max() + 1}")

        pred_clusters = {
            c.id: best_labels[id2int[c]] for index, c in enumerate(context)
        }
        pred_clusters_per_word[word] = pred_clusters

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

        if save_clusters:
            save_cluster_assignments(
                pred_clusters,
                word,
                run_path,
                sentences,
            )

        jsd[word] = compute_jsd(
            pred_clusters,
            grouping,
        )

    logging.info("returning predictions ...")
    return jsd, pred_clusters_per_word


def eval(
    get_clusters: typing.Callable,
    scores: pd.DataFrame,
    parameters: dict,
    metadata: dict,
):
    logging.info(f"eval {metadata['method']} method...")

    metadata["name_file"] = "results_testing_set"

    experiments_path = (
        f"./results/{metadata['method']}/{metadata['dataset']}/{metadata['model']}"
    )
    Path(experiments_path, "runs").mkdir(parents=True, exist_ok=True)

    results = {}

    for hyperparameters in parameters:
        run_id = _next_run_id(experiments_path)
        run_path = Path(experiments_path, "runs", run_id)

        if metadata["method"] in ["ac", "sc"]:
            jsd, _ = get_predictions(
                get_clusters,
                scores,
                hyperparameters,
                metadata=metadata,
                run_path=run_path,
            )
        else:
            jsd, _ = get_predictions_no_clusters(
                get_clusters,
                scores,
                hyperparameters,
                metadata=metadata,
                run_path=run_path,
            )

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

    data = load_data(metadata["path_to_data"])
    thresholds = (
        get_thresholds(data["prediction"])
        if metadata.get("use_threshold", False)
        else None
    )

    hyperparameter_combinations = generate_hyperparameter_combinations(
        model_hyperameter_combinations,
        metadata["normalize"],
        thresholds,
    )

    eval(
        get_clusters,
        data,
        hyperparameter_combinations,
        metadata,
    )


def grid_search_no_clusters(
    get_clusters: typing.Callable,
    model_hyperparameter_combinations: typing.List,
    metadata: dict = None,
):

    data = load_data(metadata["path_to_data"])
    thresholds = (
        get_thresholds(data["prediction"])
        if metadata.get("use_threshold", False)
        else None
    )

    hyperparameters_combinations = generate_hyperparameter_combinations(
        model_hyperparameter_combinations,
        metadata["normalize"],
        thresholds,
    )

    eval(
        get_clusters,
        data,
        hyperparameters_combinations,
        metadata,
    )
