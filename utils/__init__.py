def one_hot(labels,nb_classes):
    y_true = torch.cuda.FloatTensor(labels.size(0),nb_classes)
    y_true.zero_()
    y_true.scatter_(1, labels.data.view(-1,1), 1)
    y_true = Variable(y_true)
    return y_true