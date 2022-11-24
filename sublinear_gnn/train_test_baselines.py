import torch
import torch.nn.functional as F


def train_baselines(model, epoch_data, optimizer):
    nf_mat, conv_mat, label, train_idx = epoch_data
    model.train()

    optimizer.zero_grad()
    out = model(nf_mat, conv_mat)[train_idx].log_softmax(dim=-1)
    loss = F.nll_loss(out, label.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test_baselines(model, test_data, evaluator):
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
