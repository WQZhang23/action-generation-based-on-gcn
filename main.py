#!/usr/bin/env python
import argparse
import sys

# torchlight
import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['generation_gcn'] = import_class('processor.generation.GEN_gcn_base_Processor')
    processors['generation_attention'] = import_class('processor.generation.GEN_gcn_attention_Processor')
    #TODO: rename the processors['generation'] --> processors['generation_gcn']  ; import_class change the name
    #TODO: add the processors['generation_gcn_attention']

    # TODO: next big step --> add the args of different similarity


    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    p.start()
