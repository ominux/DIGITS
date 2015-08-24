# Copyright (c) 2015 NVIDIA CORPORATION.  All rights reserved.

from google.protobuf import text_format
try:
    import caffe_pb2
except ImportError:
    # See issue #32
    from caffe.proto import caffe_pb2

class Error(Exception):
    pass

class ValidationError(Error):
    pass

class Network(object):
    """
    """

    def __init__(self):
        self._network = None

    ### Functions for loading the network

    def load_file(self, filename):
        """
        Load a file
        """
        assert self._network is None, 'already loaded'
        with open(filename) as infile:
            text = infile.read()
            return self.load_text(text)

    def load_text(self, text):
        """
        Load some prototxt
        """
        assert self._network is None, 'already loaded'

        self._network = caffe_pb2.NetParameter()

        try:
            text_format.Merge(text, self._network)
        except text_format.ParseError as e:
            raise ValidationError(e.message)

    def requires_load(function):
        """
        Decorator ensures that a network has been loaded
        """
        def _decorator(self, *args, **kwargs):
            assert self._network is not None, 'network not loaded'
            return function(self, *args, **kwargs)
        return _decorator


    ### Functions for changing the network

    @requires_load
    def set_data_source_lmdb(self, phase, tops, filename, force=False):
        """
        Sets layer.data_param.source for a Data layer

        Arguments:
        phase -- TRAIN or TEST
        tops -- list of outputs for this layer
        filename -- location of the db

        Keyword arguments:
        force -- if False, only change the data_param.source
            if True, create the layer and delete other layers if necessary
        """
        #XXX: here
        pass

    ### Functions for retrieving the train/val/deploy network

    @requires_load
    def train_network(self):
        """
        Returns the network used during the TRAIN phase of training
        """
        train_network = caffe_pb2.NetParameter()
        for layer in self._network.layer:
            # skip layers marked for deploy
            if layer.name.startswith('deploy_'):
                continue
            # skip layers not in TRAIN phase
            if not self._layer_in_phase(layer, caffe_pb2.TRAIN):
                continue
            train_network.layer.add().CopyFrom(layer)
            # remove prefix if set
            if layer.name.startswith('train_'):
                train_network.layer[-1].name = layer.name[6:]
        return train_network

    @requires_load
    def val_network(self):
        """
        Returns the network used during the TEST phase of training
        """
        val_network = caffe_pb2.NetParameter()
        for layer in self._network.layer:
            # skip layers marked for deploy
            if layer.name.startswith('deploy_'):
                continue
            # skip layers not in TEST phase
            if not self._layer_in_phase(layer, caffe_pb2.TEST):
                continue
            val_network.layer.add().CopyFrom(layer)
            # remove prefix if set
            if layer.name.startswith('train_'):
                val_network.layer[-1].name = layer.name[6:]
        return val_network

    @requires_load
    def trainval_network(self):
        """
        Returns the network used for training
        """
        trainval_network = caffe_pb2.NetParameter()
        for layer in self._network.layer:
            # skip layers marked for deploy
            if layer.name.startswith('deploy_'):
                continue
            trainval_network.layer.add().CopyFrom(layer)
            # remove prefix if set
            if layer.name.startswith('train_'):
                trainval_network.layer[-1].name = layer.name[6:]
        return trainval_network

    @requires_load
    def deploy_network(self):
        """
        Returns the network used for deployment
        """
        deploy_network = caffe_pb2.NetParameter()
        for layer in self._network.layer:
            # skip layers marked for train
            if layer.name.startswith('train_'):
                continue
            # skip layers not in TEST phase
            if not self._layer_in_phase(layer, caffe_pb2.TEST):
                continue
            deploy_network.layer.add().CopyFrom(layer)
            # remove prefix if set
            if layer.name.startswith('deploy_'):
                deploy_network.layer[-1].name = layer.name[7:]
        return deploy_network


    ### Functions for validating the network

    @requires_load
    def validate(self):
        """
        Validate the network
        Raises Errors if something is wrong
        """
        self.validate_train_network()
        self.validate_val_network()
        self.validate_deploy_network()

    def validate_train_network(self):
        self._validate_network(self.train_network())

    def validate_val_network(self):
        self._validate_network(self.val_network())

    def validate_deploy_network(self):
        self._validate_network(self.deploy_network())

    def _validate_network(self, network):
        """
        Validates a network
        """
        if len(network.layer) == 0:
            raise ValidationError('no layers')
        bottoms = self._bottoms(network)
        tops = self._tops(network)
        for bottom in bottoms:
            if bottom not in tops and bottom not in ['data', 'label']:
                raise ValidationError('unknown layer.bottom "%s"' % bottom)

    ### Helper functions

    def _layer_in_phase(self, layer, phase):
        """
        Returns True if the layer is in the given phase
        """
        if layer.include:
            return phase in [rule.phase for rule in layer.include]
        if layer.exclude:
            return phase not in [rule.phase for rule in layer.exclude]
        return True

    def _bottoms(self, network):
        """
        Returns a unique list of bottoms for the network
        """
        bottoms = set()
        for layer in network.layer:
            for bottom in layer.bottom:
                bottoms.add(bottom)
        return bottoms

    def _tops(self, network):
        """
        Returns a unique list of tops for the network
        """
        tops = set()
        for layer in network.layer:
            for top in layer.top:
                tops.add(top)
        return tops

