#!/usr/bin/env python
#Followed the chainer chemistry tutorial for this code

from __future__ import print_function

import chainer
import numpy
import os
import pickle

from argparse import ArgumentParser
from chainer.datasets import split_dataset_random
from chainer import cuda
from chainer import functions as F
from chainer import optimizers
from chainer import training
from chainer import Variable
from chainer.iterators import SerialIterator
from chainer.training import extensions as E

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.parsers import CSVFileParser
from chainer_chemistry.dataset.preprocessors import NFPPreprocessor, GGNNPreprocessor, SchNetPreprocessor, WeaveNetPreprocessor, RSGCNPreprocessor
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.models import MLP, NFP, GGNN, SchNet, WeaveNet, RSGCN, Classifier  # NOQA
from chainer_chemistry.training.extensions import ROCAUCEvaluator


class GraphConvPredictor(chainer.Chain):
    def __init__(self, graph_conv, mlp1=None, mlp2=None, mlp3=None, mlp4=None):
        """Initializes the graph convolution predictor.

        Args:
            graph_conv: The graph convolution network required to obtain
                        molecule feature representation.
            mlp: Multi layer perceptron; used as the final fully connected
                 layer. Set it to `None` if no operation is necessary
                 after the `graph_conv` calculation.
        """

        super(GraphConvPredictor, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            if isinstance(mlp1, chainer.Link):
                self.mlp1 = mlp1
            if isinstance(mlp2, chainer.Link):
                self.mlp2 = mlp2
            if isinstance(mlp3, chainer.Link):
                self.mlp3 = mlp3
            if isinstance(mlp4, chainer.Link):
                self.mlp4 = mlp4
        if not isinstance(mlp1, chainer.Link):
            self.mlp1 = mlp1
        if not isinstance(mlp2, chainer.Link):
            self.mlp2 = mlp2
        if not isinstance(mlp3, chainer.Link):
            self.mlp3 = mlp3
        if not isinstance(mlp4, chainer.Link):
            self.mlp4 = mlp4

    def __call__(self, atoms, adjs):
        h = self.graph_conv(atoms, adjs)
        if self.mlp1:
            h = self.mlp1(h)
        if self.mlp2:
            h = self.mlp2(h)
        if self.mlp3:
            h = self.mlp3(h)
        if self.mlp4:
            h = self.mlp4(h)
        return h


def set_up_predictor(method, n_unit, conv_layers, class_num, max_atoms = 100):
    """Sets up the graph convolution network  predictor.

    Args:
        method: Method name. Currently, the supported ones are `nfp`, `ggnn`,
                `schnet`, `weavenet` and `rsgcn`.
        n_unit: Number of hidden units.
        conv_layers: Number of convolutional layers for the graph convolution
                     network.
        class_num: Number of output classes.

    Returns:
        An instance of the selected predictor.
    """

    predictor = None
    mlp1 = MLP(out_dim=class_num, hidden_dim=n_unit)
    mlp2 = MLP(out_dim=class_num, hidden_dim=n_unit)
    mlp3 = MLP(out_dim=class_num, hidden_dim=n_unit)
    mlp4 = MLP(out_dim=class_num, hidden_dim=n_unit)

    if method == 'nfp':
        print('Training an NFP predictor...')
        nfp = NFP(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers)
        predictor = GraphConvPredictor(nfp, mlp)
    elif method == 'ggnn':
        print('Training a GGNN predictor...')
        ggnn = GGNN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers)
        predictor = GraphConvPredictor(ggnn, mlp)
    elif method == 'schnet':
        print('Training an SchNet predictor...')
        schnet = SchNet(out_dim=class_num, hidden_dim=n_unit,
                        n_layers=conv_layers)
        predictor = GraphConvPredictor(schnet, None)
    elif method == 'weavenet':
        print('Training a WeaveNet predictor...')
        n_atom = max_atoms
        n_sub_layer = 1
        weave_channels = [50] * conv_layers

        weavenet = WeaveNet(weave_channels=weave_channels, hidden_dim=n_unit,
                            n_sub_layer=n_sub_layer, n_atom=n_atom)
        predictor = GraphConvPredictor(weavenet, mlp1, None, None, None)
    elif method == 'rsgcn':
        print('Training an RSGCN predictor...')
        rsgcn = RSGCN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers)
        predictor = GraphConvPredictor(rsgcn, mlp)
    else:
        raise ValueError('[ERROR] Invalid method: {}'.format(method))
    return predictor


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

    # Set up the argument parser.
    parser = ArgumentParser(description='Regression on own dataset')
    parser.add_argument('--train_datafile', '-dt', type=str,
                        default='split/cdk2/train_cdk2.csv',
                        help='csv file containing the training dataset')
    parser.add_argument('--val_datafile', '-dv', type=str,
                        default='split/cdk2/val_cdk2.csv',
                        help='csv file containing the validation dataset')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default='nfp')
    parser.add_argument('--label', '-l', nargs='+',
                        default=['value1', 'value2'],
                        help='target label for regression')
    parser.add_argument('--conv-layers', '-c', type=int, default=8,
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
    parser.add_argument('--seed', '-s', type=int, default=777,
                        help='random seed value')
    parser.add_argument('--train-data-ratio', '-r', type=float, default=0.7,
                        help='ratio of training data w.r.t the dataset')
    parser.add_argument('--protocol', type=int, default=2,
                        help='pickle protocol version')
    parser.add_argument('--model-filename', type=str, default='regressor.pkl',
                        help='saved model filename')
    parser.add_argument('--l2reg', type=float, default = 0,
                        help='weight decay for all weights')
    parser.add_argument('--max_atoms', type=int, default = 100,
                        help='number of atoms max in a molecule')
    return parser.parse_args()


def main():
    # Parse the arguments.
    args = parse_arguments()
    
    gpu = False
    if args.gpu >= 0:
        import cupy
        gpu = True

    if args.label:
        labels = args.label
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        raise ValueError('No target label was specified.')

    # Dataset preparation. Postprocessing is required for the regression task.
    def postprocess_label(label_list):
        if gpu:
            return cupy.asarray(label_list, dtype=cupy.int32)
        else:
            return numpy.asarray(label_list, dtype=numpy.int32)

    # Apply a preprocessor to the dataset.
    print('Preprocessing dataset...')
    preprocessor = set_up_preprocessor(args.method, args.max_atoms)
    parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label,
                           labels=labels, smiles_col='SMILES')
    train = parser.parse(args.train_datafile)['dataset']
    
    # Validation
    preprocessor = set_up_preprocessor(args.method, args.max_atoms)
    parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label,
                           labels=labels, smiles_col='SMILES')
    val = parser.parse(args.val_datafile)['dataset']

    # Set up the predictor.
    predictor = set_up_predictor(args.method, args.unit_num, args.conv_layers, class_num, max_atoms = args.max_atoms)
    if gpu:
        predictor = predictor.to_gpu()

    # Set up the iterator.
    train_iter = SerialIterator(train, args.batchsize)
    val_iter = SerialIterator(val, args.batchsize, repeat=False, shuffle=False)
    
    # Set up the classifier.
    classifier = Classifier(predictor,
                            lossfun=F.sigmoid_cross_entropy,
                            metrics_fun=F.binary_accuracy,
                            device=args.gpu)

    # Set up the optimizer.
    optimizer = optimizers.Adam()
    optimizer.setup(classifier)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.l2reg))

    # Set up the updater.
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                       converter=concat_mols)

    # Set up the trainer.
    print('Training...')
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    
    #metrics
    trainer.extend(E.Evaluator(val_iter, classifier))
    train_eval_iter = SerialIterator(train, args.batchsize,
                                           repeat=False, shuffle=False)
    trainer.extend(ROCAUCEvaluator(
        train_eval_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='train',
        pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # extension name='validation' is already used by `Evaluator`,
    # instead extension name `val` is used.
    trainer.extend(ROCAUCEvaluator(
        val_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='val',
        pos_labels=1, ignore_labels=-1))
    
    trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(E.PrintReport([
        'epoch', 'main/loss', 'main/accuracy', 'train/main/roc_auc',
        'validation/main/loss', 'validation/main/accuracy',
        'val/main/roc_auc', 'elapsed_time']))
    trainer.extend(E.ProgressBar())
    trainer.run()

    # Save the classifier's parameters.
    model_path = os.path.join(args.out, args.model_filename)
    print('Saving the trained model to {}...'.format(model_path))
    classifier.save_pickle(model_path, protocol=args.protocol)


if __name__ == '__main__':
    main()
