import torch
from collections import defaultdict

class TensorList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, tensor):
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.long, device=device)

        self.tensor = tensor
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, tensor):
        for k, v in tensor.extra_fields.items():
            self.extra_fields[k] = v

    # Tensor-like methods

    def to(self, device):
        tensor = TensorList(self.tensor.to(device))
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
                tensor.add_field(k, v)
            else:
                tensor.add_field(k, v)
        return tensor

    def __getitem__(self, item):
        tensor = TensorList(self.tensor[item])
        for k, v in self.extra_fields.items():
            if isinstance(v, torch.Tensor) or isinstance(item, int):
                tensor.add_field(k, v[item])
            else:
                tensor.add_field(k, [v[ind] for ind, i in enumerate(item) if i == 1])
        return tensor

    def __len__(self):
        return self.tensor.shape[0]

    def copy_with_fields(self, fields, skip_missing=False):
        tensor = TensorList(self.tensor)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                tensor.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return tensor

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_sentences={}, ".format(len(self))
        return s

    def trim(self):
        self.tensor = self.tensor[:, torch.sum(self.tensor, 0) > 0]

    def get_target(self):
        target = self.tensor[:, 1:].clone().detach()
        return target

def to_tensor_list(tensors, size_divisible=0):
    """
    tensors can be an TensorList, a torch.Tensor or
    an iterable of Tensors or TensorList. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the beginning of the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, TensorList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        return TensorList(tensors)
    elif isinstance(tensors, (tuple, list)):
        if isinstance(tensors[0], TensorList):
            max_size = max([t.tensor.shape[1] for t in tensors])
            batch_shape = sum([t.tensor.shape[0] for t in tensors])
            extra_fields = defaultdict(list)
            for t in tensors:
                for key,value in t.extra_fields.items():
                    extra_fields[key].extend(value)
            tensors = [t.tensor for t in tensors]
        elif isinstance(tensors[0], torch.tensor):
            max_size = max([t.shape[1] for t in tensors])
            batch_shape = sum([t.shape[0] for t in tensors])
            extra_fields = {}
        else:
            raise TypeError("Unsupported type for to_tensor_list: {}".format(type(tensors)))

        new_t = torch.zeros((batch_shape, max_size), dtype=torch.long, device=tensors[0].device)
        w = 0
        for t in tensors:
            new_t[w:w+t.shape[0], -t.shape[1]:].copy_(t)
            w += t.shape[0]

        output = TensorList(new_t)
        for key,value in extra_fields.items():
            output.add_field(key, value)

        return output
    else:
        raise TypeError("Unsupported type for to_tensor_list: {}".format(type(tensors)))

if __name__ == "__main__":
    bbox = TensorList([[0, 0, 10, 10], [0, 0, 5, 5]])
