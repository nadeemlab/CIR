import numpy as np


def jaccard_index(target, pred, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, num_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred >= cls
        target_inds = target >= cls

        intersection = pred_inds[target_inds].long().sum(
        ).data.cpu()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu() + \
            target_inds.long().sum().data.cpu() - intersection
        if union == 0:
            # If there is no ground truth, do not include in evaluation
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(union))
    return np.array(ious)
