import torch
import torch.nn.functional as F


def KL(mu1, sig1, mu2, sig2):
    return torch.log(sig2 / sig1 +
                     1e-5) + (sig1**2 +
                              (mu1 - mu2)**2) / (2 * sig2**2 + 1e-5) - 0.5


def maha_dist(v1, s1, v2, s2):
    # assume v1 and v2 belong to same distribution, so s1=s2
    d = (v1-v2)**2
    d /= s2**2
    d = torch.sum(d)
    return torch.sqrt(d).view(1)


def maha_cos(v1, s1, v2, s2):
    # assume v1 and v2 belong to same distribution, so s1=s2
    # https://www.wseas.org/multimedia/journals/signal/2016/a425814-486.pdf
    u1 = v1/s1
    u2 = v2/s2
    return F.cosine_similarity(u1, u2)


def arith_mean(tensors):
    return sum(tensors)/len(tensors)


def f1(preds, labels, threshold=None):
    if threshold is not None:
        preds = 1*(preds > threshold)

    tp = torch.sum(preds * labels)
    fp = torch.sum(preds * (1-labels))
    # tn = torch.sum((1-preds) * (1-labels))
    fn = torch.sum((1-preds) * labels)

    return tp/(tp + 0.5*(fp+fn))


def acc(preds, labels, threshold=None):
    if threshold is not None:
        preds = 1*(preds > threshold)

    return torch.sum(preds * labels)/labels.nelement()
