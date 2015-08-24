# Copyright (c) 2015 NVIDIA CORPORATION.  All rights reserved.

import os.path
import unittest

from nose.tools import raises

import digits.model.images.classification.test_views
import digits.model.images.generic.test_views

from . import caffe_network as _

class TestNetwork(object):
    def test_classification_test_network(self):
        network = _.Network()
        network.load_text(
                digits.model.images.classification.test_views.BaseViewsTest.CAFFE_NETWORK)
        network.validate()

    def test_generic_test_network(self):
        network = _.Network()
        network.load_text(
                digits.model.images.generic.test_views.BaseViewsTest.CAFFE_NETWORK)
        network.validate()

    def test_standard_networks(self):
        for filename in [
                'lenet.prototxt',
                'alexnet.prototxt',
                'googlenet.prototxt',
                ]:
            yield self.check_standard_network, filename

    def check_standard_network(self, filename):
        network = _.Network()
        network.load_file(
                os.path.join(
                    os.path.dirname(digits.__file__),
                    'standard-networks', filename)
                )
        network.validate()

    @raises(_.ValidationError)
    def test_blank(self):
        network = _.Network()
        network.load_text('')
        network.validate()

    @raises(_.ValidationError)
    def test_bad_format(self):
        network = _.Network()
        network.load_text('this-is-not-prototxt')
        network.validate()

    @raises(_.ValidationError)
    def test_bad_param(self):
        network = _.Network()
        network.load_text('layer {name: "data"; type: "Data"; not_a_param: 1}')
        network.validate()

    # Requires https://github.com/BVLC/caffe/pull/2930
    @unittest.skip('LayerTypeList not exposed yet')
    @raises(_.ValidationError)
    def test_bad_layer_type(self):
        network = _.Network()
        network.load_text('layer {type: "not-a-layer-type"}')
        network.validate()

    @raises(_.ValidationError)
    def test_disconnected_deploy_network(self):
        network = _.Network()
        network.load_text("""
        layer {
            name:'1'
            type:'Data'
            top:'1'
        }
        layer {
            name:'train_2'
            type:'InnerProduct'
            bottom:'1'
            top:'2'
        }
        layer {
            name:'3'
            type:'InnerProduct'
            bottom:'2'
            top:'3'
        }""")
        network.validate()

