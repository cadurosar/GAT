import torch

def one_hot(labels,nb_classes):
    y_true = torch.cuda.FloatTensor(labels.size(0),nb_classes)
    y_true.zero_()
    y_true.scatter_(1, labels.data.view(-1,1), 1)
    y_true = torch.autograd.Variable(y_true)
    return y_true

class ListModule(torch.nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)