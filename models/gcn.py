import chanier

class GraphConvPredictor(chainer.Chain):
    def __init__(self, graph_conv, mlp=None):
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
            if isinstance(mlp, chainer.Link):
                self.mlp = mlp
        if not isinstance(mlp, chainer.Link):
            self.mlp = mlp

    def __call__(self, atoms, adjs):
        h = self.graph_conv(atoms, adjs)
        if self.mlp:
            h = self.mlp(h)
        return h