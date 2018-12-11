import tensorflow as tf
import torch
import utils


class BaseGAttN(torch.nn.Module):
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = (utils.one_hot(labels,nb_classes)*class_weights).sum(dim=-1)
        xentropy = torch.nn.functional.cross_entropy(input=logits,target=labels,reduction="none")
        return (xentropy*sample_wts).mean()

##########################
# Adapted from tkipf/gcn #
##########################

    def masked_softmax_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        _,labels = torch.max(labels,dim=1)
        loss = torch.nn.functional.cross_entropy(input=logits, target=labels,reduction="none")
        mask = mask/mask.mean()
        loss = loss*mask
        return loss.mean()

    def masked_accuracy(logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = torch.eq(torch.argmax(logits,1),torch.argmax(labels,1))
        accuracy_all = correct_prediction.float()
        mask = mask.float()
        mask /= mask.mean()
        accuracy_all *= mask
        return accuracy_all.mean()