import numpy as np
import math
from micrograd.modules import Value
from typing import List
from pydantic import BaseModel, validate_arguments,ValidationError

@validate_arguments(config = dict(arbitrary_types_allowed = True))
def numerical_stability(outputs:List[Value]):
    max_val = 0
    for val in outputs:
        if val.data >= max_val:
            max_val = val.data
    stablized_activations = [val - max_val for val in outputs]
    return stablized_activations


@validate_arguments(config = dict(arbitrary_types_allowed = True))
def softmax(outputs:List[List[Value]],stablize:bool = True):
    sums = list()
    outputs = [numerical_stability(instance) for instance in outputs] if stablize else outputs
    # calculate exponent for each activation from each training instance
    # Each activation is the output from a neuron
    exponent_list = [[val_obj.exp() for val_obj in instance] for instance in outputs]

    #sum up the exponent values to get denominator
    for activations in exponent_list:
        sums.append(sum(activations))

    #calculate softmax probability of each activation 
    softmax_probabilities = [[val/j for val in exponent_list[i]] for i,j in enumerate(sums)]
    return softmax_probabilities

@validate_arguments(config = dict(arbitrary_types_allowed = True))
def cross_entropy_loss_logits(outputs:List[List[Value]],classes = List[int],stablize:bool = True,reduction:str = 'mean'):
    softmax_probabilities = softmax(outputs,stablize)

    #select the probability of the appropriate class
    assert len(outputs) == len(classes),"Length mismatch"
    pred_class_probabilities = [i[j] for i,j in zip(softmax_probabilities,classes)]

    #find the natural logarithm of each probability
    loss = sum((-1* val.ln() for val in pred_class_probabilities))

    assert reduction in ['mean','add'],"Invalid reduction parameter"
    loss = loss/len(outputs) if reduction == 'mean' else loss
    
    return loss