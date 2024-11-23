from sklearn.metrics import roc_auc_score

class AUCMetric:
    def __init__(self):
        pass

    def compute_auc(self, true_labels: list, scores: list) -> float:
        return roc_auc_score(true_labels, scores)