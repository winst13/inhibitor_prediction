#!/usr/bin/env python

from __future__ import print_function

import json
import numpy
import os
import pickle

from argparse import ArgumentParser
from chainer.iterators import SerialIterator
from chainer.training.extensions import Evaluator

from chainer import cuda
from chainer import Variable
from chainer import functions as F

from chainer_chemistry.models.prediction import Classifier
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.parsers import CSVFileParser
from chainer_chemistry.dataset.preprocessors import NFPPreprocessor, GGNNPreprocessor, SchNetPreprocessor, WeaveNetPreprocessor, RSGCNPreprocessor

# These imports are necessary for pickle to work.
from train_own_dataset import GraphConvPredictor  # NOQA

def set_up_preprocessor(method, max_atoms):
    preprocessor = None

    if method == 'nfp':
        preprocessor = NFPPreprocessor(max_atoms = max_atoms)
    elif method == 'ggnn':
        preprocessor = GGNNPreprocessor(max_atoms = max_atoms)
    elif method == 'schnet':
        preprocessor = SchNetPreprocessor(max_atoms = max_atoms)
    elif method == 'weavenet':
        preprocessor = WeaveNetPreprocessor(max_atoms = max_atoms)
    elif method == 'rsgcn':
        preprocessor = RSGCNPreprocessor(max_atoms = max_atoms)
    else:
        raise ValueError('[ERROR] Invalid method: {}'.format(method))
    return preprocessor

def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn']
    kinase_list = ['cdk2', 'egfr_erbB1', 'gsk3b', 'hgfr', 'map_k_p38a', 'tpk_lck', 'tpk_src', 'vegfr2']

    # Set up the argument parser.
    parser = ArgumentParser(description='Regression on own dataset')
    parser.add_argument('--datafile', '-d', type=str, choices=kinase_list,
                        help='csv file containing the dataset', default='cdk2')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default='nfp')
    parser.add_argument('--label', '-l', nargs='+',
                        default='inhibits',
                        help='target label for regression')
    parser.add_argument('--conv-layers', '-c', type=int, default=4,
                        help='number of convolution layers')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='batch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='id of gpu to use; negative value means running'
                        'the code on cpu')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to save the computed model to')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--unit-num', '-u', type=int, default=16,
                        help='number of units in one layer of the model')
    parser.add_argument('--protocol', type=int, default=2,
                        help='pickle protocol version')
    parser.add_argument('--in-dir', '-i', type=str, default='result',
                        help='directory containing the saved model')
    parser.add_argument('--model-filename', type=str, default='classifier.pkl',
                        help='saved model filename')
    parser.add_argument('--max_atoms', type=int, default = 100,
                        help='number of atoms max in a molecule')
    return parser.parse_args()


def main():
    # Parse the arguments.
    args = parse_arguments()

    if args.label:
        labels = args.label
    else:
        raise ValueError('No target label was specified.')

    # Dataset preparation.
    def postprocess_label(label_list):
        return numpy.asarray(label_list, dtype=numpy.int32)

    print('Preprocessing dataset...')
    train_path = "split/" + args.datafile + "/train_" + args.datafile + ".csv"
    val_path = "split/" + args.datafile + "/val_" + args.datafile + ".csv"
    test_path = "split/" + args.datafile + "/test_" + args.datafile + ".csv"
    
    preprocessor = set_up_preprocessor(args.method, args.max_atoms)
    parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label,
                           labels=labels, smiles_col='SMILES')
    train = parser.parse(train_path)['dataset']
    val = parser.parse(val_path)['dataset']
    test = parser.parse(test_path)['dataset']

    print('Predicting...')
    # Set up the regressor.
    model_path = os.path.join(args.in_dir, args.model_filename)
    classifier = Classifier.load_pickle(model_path, device=args.gpu)

    # Perform the prediction.
    print('Evaluating...')
    train_iterator = SerialIterator(train, 1, repeat=False, shuffle=False)
    val_iterator = SerialIterator(val, 1, repeat=False, shuffle=False)
    test_iterator = SerialIterator(test, 1, repeat=False, shuffle=False)
    
    
    with open("result/" + args.datafile + "/train_pred_" + args.datafile + ".csv", "w+") as file:
        for i in train_iterator:
            pred = F.sigmoid(classifier.predictor(numpy.asarray([i[0][0]]), numpy.asarray([i[0][1]])))
            file.write("%f\n"%(pred._data[0][0][0]))
    
    with open("result/" + args.datafile + "/val_pred_" + args.datafile + ".csv", "w+") as file:
        for i in val_iterator:
            pred = F.sigmoid(classifier.predictor(numpy.asarray([i[0][0]]), numpy.asarray([i[0][1]])))
            file.write("%f\n"%(pred._data[0][0][0]))
           
    with open("result/" + args.datafile + "/test_pred_" + args.datafile + ".csv", "w+") as file:
        for i in test_iterator:
            pred = F.sigmoid(classifier.predictor(numpy.asarray([i[0][0]]), numpy.asarray([i[0][1]])))
            file.write("%f\n"%(pred._data[0][0][0]))
    
    
    eval_result = Evaluator(test_iterator, classifier, converter=concat_mols,
                            device=args.gpu)()

    # Prevents the loss function from becoming a cupy.core.core.ndarray object
    # when using the GPU. This hack will be removed as soon as the cause of
    # the issue is found and properly fixed.
    loss = numpy.asscalar(cuda.to_cpu(eval_result['main/loss']))
    eval_result['main/loss'] = loss
    print('Evaluation result: ', eval_result)

    with open(os.path.join(args.in_dir, 'eval_result.json'), 'w') as f:
        json.dump(eval_result, f)


if __name__ == '__main__':
    main()
