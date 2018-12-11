import tensorflow as tf
import torch
import utils


class BaseGAttN:
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = (utils.one_hot(labels,nb_classes)*class_weights).sum(dim=-1)
        xentropy = torch.nn.functional.cross_entropy(logits,labels,weight=sample_wts)
        return xentropy

##########################
# Adapted from tkipf/gcn #
##########################

    def masked_softmax_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        return torch.nn.functional.cross_entropy(logits,labels,weight=mask)

    def masked_accuracy(logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = torch.eq(torch.argmax(logits,1),torch.argmax(labels,1))
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = correct_prediction.float()
        mask = mask.float()
        mask /= mask.mean()
        accuracy_all *= mask
        return accuracy_all.mean()