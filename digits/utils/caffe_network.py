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

class RequiresForceError(Error):
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
        phase -- train or val
        tops -- list of outputs for this layer
        filename -- location of the db

        Keyword arguments:
        force -- if False, only change the data_param.source
            if True, create the layer and delete other layers if necessary
        """
        old_data_layers = [(i,l) for i,l in enumerate(self._network.layer)
                if Network._is_data_layer(l) and self._layer_in_phase(l, phase)]

        needs_new_layer = False

        # no layer exists
        if len(old_data_layers) == 0:
            needs_new_layer = True

        # one layer found
        elif len(old_data_layers) == 1:
            old_layer_index, old_layer = old_data_layers[0]

            needs_delete = False

            # check layer.type
            if old_layer.type != 'Data':
                if not force:
                    raise RequiresForceError('data layer is type "%s"' % old_layer.type)
                needs_delete = True
            else:
                # check layer.top
                if len(old_layer.top) != len(tops):
                    if not force:
                        raise RequiresForceError('data layer has too many tops')
                    needs_delete = True
                else:
                    for old_top, new_top in zip(old_layer.top, tops):
                        if old_top != new_top:
                            if not force:
                                raise RequiresForceError('tops do not match (%s vs %s)' % (old_top, new_top))
                            needs_delete = True
                            break

            # delete the layer
            if needs_delete:
                print 'Deleting layer name="%s", type="%s", top="%s"' % (
                        old_layer.name, old_layer.type, old_layer.top)
                del self._network.layer[old_layer_index]
                needs_new_layer = True

        # multiple layers
        elif len(old_data_layers) > 1:
            if not force:
                raise RequiresForceError('multiple data layers found')
            for old_layer_index, old_layer in old_data_layers:
                print 'Deleting layer name="%s", type="%s", top="%s"' % (
                        old_layer.name, old_layer.type, old_layer.top)
                del self._network.layer[old_layer_index]
                needs_new_layer = True

        if needs_new_layer:
            data_layer = caffe_pb2.LayerParameter(
                    name=','.join(tops),
                    type='Data',
                    )
            for top in tops:
                data_layer.top.append(top)
            self._set_layer_phases(data_layer, phase)
            data_layer.data_param.source = filename
            self._insert_layer_front(self._network, data_layer)


    ### Functions for retrieving the train/val/deploy network

    @requires_load
    def train_network(self):
        """
        Returns the network used during the TRAIN phase of training
        """
        train_network = caffe_pb2.NetParameter()
        for layer in self._network.layer:
            # skip layers not in train phase
            if not self._layer_in_phase(layer, 'train'):
                continue
            train_network.layer.add().CopyFrom(layer)
            self._cleanup_layer_name(train_network.layer[-1])
        return train_network

    @requires_load
    def val_network(self):
        """
        Returns the network used during the TEST phase of training
        """
        val_network = caffe_pb2.NetParameter()
        for layer in self._network.layer:
            # skip layers not in val phase
            if not self._layer_in_phase(layer, 'val'):
                continue
            val_network.layer.add().CopyFrom(layer)
            self._cleanup_layer_name(val_network.layer[-1])
        return val_network

    @requires_load
    def trainval_network(self):
        """
        Returns the network used for training
        """
        trainval_network = caffe_pb2.NetParameter()
        for layer in self._network.layer:
            # skip layers not in train or val phase
            if not self._layer_in_phase(layer, 'train') and \
                    not self._layer_in_phase(layer, 'val'):
                continue
            trainval_network.layer.add().CopyFrom(layer)
            self._cleanup_layer_name(trainval_network.layer[-1])
        return trainval_network

    @requires_load
    def deploy_network(self):
        """
        Returns the network used for deployment
        """
        deploy_network = caffe_pb2.NetParameter()
        for layer in self._network.layer:
            # skip layers not in deploy phase
            if not self._layer_in_phase(layer, 'deploy'):
                continue
            deploy_network.layer.add().CopyFrom(layer)
            self._cleanup_layer_name(deploy_network.layer[-1])
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

    @staticmethod
    def _is_data_layer(layer):
        """
        Returns True if the layer is a data layer
        """
        return 'data' in layer.type.lower()

    @classmethod
    def _layer_in_phase(cls, layer, phase):
        """
        Returns True if the layer is in the given phase (train/val/deploy)
        """
        if phase == 'train':
            if layer.name.startswith('deploy_'):
                return False
            return cls._layer_in_caffe_phase(layer, caffe_pb2.TRAIN)
        elif phase == 'val':
            if layer.name.startswith('deploy_'):
                return False
            return cls._layer_in_caffe_phase(layer, caffe_pb2.TEST)
        elif phase == 'deploy':
            if layer.name.startswith('train_'):
                return False
            return cls._layer_in_caffe_phase(layer, caffe_pb2.TEST)
        else:
            raise ValueError('invalid phase')

    @staticmethod
    def _layer_in_caffe_phase(layer, phase):
        """
        Returns True if the layer is in the given phase (TRAIN/TEST)
        """
        if layer.include:
            return phase in [rule.phase for rule in layer.include]
        if layer.exclude:
            return phase not in [rule.phase for rule in layer.exclude]
        return True

    @classmethod
    def _set_layer_phases(cls, layer, phases):
        """
        Sets the layer to only be included in the given phases
        Options: all,train,trainval,val,deploy
        """
        # include in all
        cls._cleanup_layer_name(layer)
        layer.ClearField('exclude')
        layer.ClearField('include')

        if phases == 'all':
            return
        elif phases == 'trainval':
            if not cls._is_data_layer(layer):
                layer.name = 'train_%s' % layer.name
        elif phases == 'train':
            if not cls._is_data_layer(layer):
                layer.name = 'train_%s' % layer.name
            layer.include.add(phase = caffe_pb2.TRAIN)
        elif phases == 'val':
            if not cls._is_data_layer(layer):
                layer.name = 'train_%s' % layer.name
            layer.include.add(phase = caffe_pb2.TEST)
        elif phases == 'deploy':
            if cls._is_data_layer(layer):
                raise ValueError("can't add data layer to deploy network")
            layer.name = 'deploy_%s' % layer.name
        else:
            raise ValueError('invalid phases')

    @staticmethod
    def _cleanup_layer_name(layer):
        """
        Remove prefix from layer name if exists
        """
        if layer.name.startswith('train_'):
            layer.name = layer.name[6:]
        if layer.name.startswith('deploy_'):
            layer.name = layer.name[7:]

    @staticmethod
    def _insert_layer_front(network, layer):
        """
        Add a layer to the front of the network
        """
        old_layers = [l for l in network.layer]
        network.ClearField('layer')
        network.layer.add().CopyFrom(layer)
        for l in old_layers:
            network.layer.add().CopyFrom(l)

    @staticmethod
    def _bottoms(network):
        """
        Returns a unique list of bottoms for the network
        """
        bottoms = set()
        for layer in network.layer:
            for bottom in layer.bottom:
                bottoms.add(bottom)
        return bottoms

    @staticmethod
    def _tops(network):
        """
        Returns a unique list of tops for the network
        """
        tops = set()
        for layer in network.layer:
            for top in layer.top:
                tops.add(top)
        return tops

