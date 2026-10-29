
import re

from mmengine.logging import print_log

def patten_match(name, patten):
    """
    support regular expression and suffix match

    patten_match("model.attn.q", ".*q")
    :param name:
    :param patten:
    :return:
    """
    return bool(re.fullmatch(patten, name))



def unfreeze_param(named_parameters, unfreeze):
    for k, param in named_parameters:
        if any([patten_match(k, p) for p in unfreeze]):
            if not param.requires_grad:
                print_log(f"Unfreeze param {k}", logger='current')
                param.requires_grad = True
            else:
                print_log(f"The param {k} is not frozon", logger='current')