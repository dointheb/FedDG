import copy
import torch
from typing import  Dict, List

def static_avg(weights:List[Dict[str,torch.Tensor]]) -> Dict[str,torch.Tensor]:

    weights_avg = copy.deepcopy(weights[0])
    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))
    
    return weights_avg

def dynamic_avg(weights:List[Dict[str,torch.Tensor]],cls_res) -> Dict[str,torch.Tensor]:
    cls_weights = cls_res.view(-1)
    weights_avg = {key: torch.zeros_like(weights[0][key]) for key in weights[0].keys()}
    for i, model_weights in enumerate(weights):
        for key in weights_avg.keys():
            weights_avg[key] += model_weights[key] * cls_weights[i]

    return weights_avg