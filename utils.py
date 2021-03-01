import torch
import dgl
import random
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(pathname)s | line %(lineno)d\n%(levelname)s: %(message)s')


def backup_code(log_path):
    os.system(F'rsync -avr --exclude-from=".gitignore" * {log_path}/codebackup')


def count_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def write_dict(d, file, num_tab=0):
    for k, v in d.items():
        if not isinstance(v, dict):
            file.write('{}{}: {}\n'.format(num_tab * '\t' + '', k, v))
        else:
            file.write('{}{}:\n'.format(num_tab * '\t' + '', k))
            write_dict(v, file, num_tab + 1)
