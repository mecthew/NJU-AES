#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/11/20 21:10
# @Author:  Mecthew

from integration.ingestion import do_ingestion
from modeling.preprocess import compute_string_kernel
from modeling.constant import ROOT_DIR
import sys
import os


if __name__ == '__main__':
    integration_dir = os.path.join(ROOT_DIR, 'integration')
    modeling_dir = os.path.join(ROOT_DIR, 'modeling')
    sys.path.append(integration_dir)
    sys.path.append(modeling_dir)
    do_ingestion()
    # compute_string_kernel()
