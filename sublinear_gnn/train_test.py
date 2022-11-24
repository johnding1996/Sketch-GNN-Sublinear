import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def train(model, epoch_data, optimizer, clip_threshold, num_sketches):
    nf_sketches, conv_sketches, ll_cs_list, label, train_idx = epoch_data
    model.train()

    optimizer.zero_grad()

    outs = []
    for i in range(num_sketches):
        out_sketches = model(nf_sketches[i], conv_sketches[i])
        outs.extend([cs.unsketch_mat(os) for cs, os in zip(ll_cs_list[i], out_sketches)])
    out = torch.median(torch.stack(outs, dim=0), dim=0).values
    out = out[train_idx].log_softmax(dim=-1)
    loss = F.nll_loss(out, label.squeeze(1)[train_idx])
    loss.backward()

    if clip_threshold is not None:
        clip_grad_norm_(model.parameters(), clip_threshold)
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, test_data, evaluator):
    nf_mat, conv_mat, label, train_idx, valid_idx, test_idx = test_data
    model.eval()

    out = model(nf_mat, conv_mat).log_softmax(dim=-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': label[train_idx],
        'y_pred': y_pred[train_idx],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': label[valid_idx],
        'y_pred': y_pred[valid_idx],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': label[test_idx],
        'y_pred': y_pred[test_idx],
    })['acc']

    return train_acc, valid_acc, test_acc
