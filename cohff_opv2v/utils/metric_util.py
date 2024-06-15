
import numpy as np
from mmseg.utils import get_root_logger
import torch
import torch.distributed as dist

import json
from json import JSONEncoder
import pickle
import os

json_types = (list, dict, str, int, float, bool, type(None))


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


class MeanIoU:

    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name
        ):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes).cuda()
        self.total_correct = torch.zeros(self.num_classes).cuda()
        self.total_positive = torch.zeros(self.num_classes).cuda()

    def _after_step(self, outputs, targets):
        outputs = outputs[targets != self.ignore_label].cpu()
        targets = targets[targets != self.ignore_label].cpu()

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(targets == c).item()
            self.total_correct[i] += torch.sum((targets == c)
                                               & (outputs == c)).item()
            self.total_positive[i] += torch.sum(outputs == c).item()

    def _after_epoch(self, res_dir, res_name='result_iou.json'):
        result_iou = {}
        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou.item())

        miou = np.mean(ious)
        logger = get_root_logger()
        logger.info(f'Validation per class iou {self.name}:')
        for iou, label_str in zip(ious, self.label_str):
            result_iou[label_str] = iou
            logger.info('%s : %.2f%%' % (label_str, iou * 100))
            


        with open(os.path.join(res_dir, res_name), "w") as jsfile:
            json.dump(result_iou, jsfile)
        
        return miou * 100
        
