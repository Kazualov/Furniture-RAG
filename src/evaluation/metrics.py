import math
def recall_at_k(relevant, predicted, k):
    predicted = predicted[:k]

    hits = len(set(relevant) & set(predicted))

    if not relevant:
        return 0

    return hits / len(relevant)
def reciprocal_rank(relevant, predicted):
    for i, pid in enumerate(predicted):
        if pid in relevant:
            return 1 / (i + 1)

    return 0
def dcg(relevant, predicted, k):
    score = 0

    for i, pid in enumerate(predicted[:k]):
        if pid in relevant:
            score += 1 / math.log2(i + 2)

    return score
def ndcg(relevant, predicted, k):
    ideal = dcg(relevant, relevant, min(k, len(relevant)))

    if ideal == 0:
        return 0

    return dcg(relevant, predicted, k) / ideal