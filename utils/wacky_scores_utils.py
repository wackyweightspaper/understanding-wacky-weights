import numpy as np
import matplotlib.pyplot as plt


def sort_and_get_top_n(wacky_scores, tokenizer, n=100, threshold=3, normalize=False):
    wacky_scores_sorted = [(tokenizer.convert_ids_to_tokens(int(token)), np.mean(scores)) for token, scores in wacky_scores.items() if len(scores) > threshold]
    wacky_scores_sorted = sorted(wacky_scores_sorted, key=lambda x: x[1], reverse=False)
    
    if normalize and wacky_scores_sorted:
        max_score = max(score for _, score in wacky_scores_sorted)
        if max_score > 0:
            wacky_scores_sorted = [(token, score / max_score) for token, score in wacky_scores_sorted]

    if n is not None: 
        return wacky_scores_sorted[:n]
    return wacky_scores_sorted

def sort_and_get_top_n_classic(wacky_scores, n=100, threshold=3, normalize=False):
    wacky_scores_sorted = [(token, np.mean(scores)) for token, scores in wacky_scores.items() if len(scores) > threshold]
    wacky_scores_sorted = sorted(wacky_scores_sorted, key=lambda x: x[1], reverse=False)
    
    if normalize and wacky_scores_sorted:
        max_score = max(score for _, score in wacky_scores_sorted)
        if max_score > 0:
            wacky_scores_sorted = [(token, score / max_score) for token, score in wacky_scores_sorted]

    if n is not None: 
        return wacky_scores_sorted[:n]
    return wacky_scores_sorted


def plot_wackiness(model_curves):
    labels = list(model_curves.keys())
    values = [model_curves[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, series in zip(labels, values):
        curve = [score for _, score in series]
        ax.plot(curve, linewidth=1.5, label=label)

    ax.set_xlabel('Token rank (high -> low wackiness)')
    ax.set_ylabel('Non-wackiness score')
    ax.set_title('Comparison of Token Non-Wackiness Across Models')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def chunk_non_wackiness(series, n_chunks=1000):
    scores = np.array([score for _, score in series], dtype=float)
    if scores.size == 0:
        return []

    edges = np.linspace(0, scores.size, num=n_chunks + 1, dtype=int)
    chunked_avgs = [
        scores[edges[i]:edges[i + 1]].mean()
        for i in range(n_chunks)
        if edges[i] < edges[i + 1]
    ]
    return chunked_avgs


def chunk_wackiness(series, n_chunks=1000):
    scores = np.array([1 - score for _, score in series], dtype=float)
    if scores.size == 0:
        return []

    edges = np.linspace(0, scores.size, num=n_chunks + 1, dtype=int)
    chunked_avgs = [
        scores[edges[i]:edges[i + 1]].mean()
        for i in range(n_chunks)
        if edges[i] < edges[i + 1]
    ]
    return chunked_avgs

def chunk_curve_auc(chunk_curves, normalization_constant=1000):
    """
    chunk_curves: dict[name -> list/array of chunk averages]
    Returns: dict[name -> area under curve]
    """
    aucs = {}
    for label, averages in chunk_curves.items():
        if not averages:
            aucs[label] = 0.0
            continue
        x = np.arange(1, len(averages) + 1)
        if normalization_constant is not None:
            aucs[label] = np.trapezoid(averages, x) / normalization_constant
        else:
            aucs[label] = np.trapezoid(averages, x)
    return aucs