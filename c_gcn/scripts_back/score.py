import torch


def compute_answers_scores(answers_indices):
    """Generate VQA based answer scores for answers_indices.

    Args:
        answers_indices (torch.LongTensor): tensor containing indices of the answers

    Returns:
        torch.FloatTensor: tensor containing scores.

    """
    scores = torch.zeros(15, dtype=torch.float)
    gt_answers = list(enumerate(answers_indices))
    unique_answers = set(answers_indices.tolist())

    for answer in unique_answers:
        accs = []
        for gt_answer in gt_answers:
            other_answers = [item for item in gt_answers if item != gt_answer]

            matching_answers = [item for item in other_answers if item[1] == answer]
            acc = min(1, float(len(matching_answers)) / 3)
            accs.append(acc)
        avg_acc = sum(accs) / len(accs)

        if answer != 0:
            scores[answer] = avg_acc

    return scores


a_indices = torch.Tensor([7, 7, 7, 3, 3, 3, 3, 0, 0, 0]).long()
compute_answers_scores(a_indices)