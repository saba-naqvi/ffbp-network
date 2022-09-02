"""
Assignment 5
by Saba Naqvi
03/15/2022
This program creates, implements, trains and tests a FeedForward
BackPropagation neural network on three different datasets.
"""

from enum import Enum
from abc import ABC, abstractmethod
from random import random
import numpy as np
from collections import deque
from random import sample, shuffle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataMismatchError(Exception):
    """ Raised when number of labels != number of features in NNData. """
    pass


class NNData:
    """ Maintain and dispense examples for use by neural network. """

    class Order(Enum):
        """ Indicate whether data will be shuffled for each new epoch. """
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        """ Indicate whether data is for training or testing. """
        TRAIN = 0
        TEST = 1

    @staticmethod
    def percentage_limiter(factor):
        """ Ensure that percentage is bounded between 0 and 1. """
        return min(1, max(factor, 0))

    def __init__(self, features=None, labels=None, train_factor=.9):
        """ Initialize attributes of NNData class. """
        self._train_factor = NNData.percentage_limiter(train_factor)
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._features = None
        self._labels = None
        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()
        try:
            self.load_data(features, labels)
        except (ValueError, DataMismatchError):
            pass

    def load_data(self, features=None, labels=None):
        """ Create numpy arrays for features and labels data, make call
        to self.split_set().
        """
        if features is None or labels is None:
            self._features = None
            self._labels = None
            return
        if len(features) != len(labels):
            self._features = None
            self._labels = None
            raise DataMismatchError
        try:
            self._features = np.array(features, dtype="f")
            self._labels = np.array(labels, dtype="f")
        except ValueError:
            self._features = None
            self._labels = None
            raise ValueError
        self.split_set()

    def split_set(self, new_train_factor=None):
        """ Create indirect indices lists for training and testing examples. """
        if new_train_factor is not None:
            self._train_factor = self.percentage_limiter(new_train_factor)
        if self._features is None:
            self._train_indices = []
            self._test_indices = []
        number_of_examples = len(self._features)
        training_examples_count = round(self._train_factor * number_of_examples)
        indices_list = list(range(number_of_examples))
        self._train_indices = sample(indices_list, k=training_examples_count)
        self._test_indices = [index_value for index_value in indices_list
                              if index_value not in self._train_indices]
        shuffle(self._train_indices)
        shuffle(self._test_indices)

    def prime_data(self, target_set=None, order=None):
        """ Populate self._train_pool and self._test_pool with indices
        from self._train_indices and self._test_indices.
        """
        train_pool = deque(self._train_indices)
        test_pool = deque(self._test_indices)
        if order == NNData.Order.SEQUENTIAL or order is None:
            pass
        if order == NNData.Order.RANDOM:
            shuffle(train_pool)
            shuffle(test_pool)
        if target_set == NNData.Set.TRAIN:
            self._train_pool = train_pool
        elif target_set == NNData.Set.TEST:
            self._test_pool = test_pool
        elif target_set is None:
            self._train_pool = train_pool
            self._test_pool = test_pool

    def get_one_item(self, target_set=None):
        """ Return tuple of one example from desired target_set. """
        if target_set == NNData.Set.TRAIN or target_set is None:
            if len(self._train_pool) == 0:
                return None
            pool_index = self._train_pool.popleft()
        elif target_set == NNData.Set.TEST:
            if len(self._test_pool) == 0:
                return None
            pool_index = self._test_pool.popleft()
        return self._features[pool_index], self._labels[pool_index]

    def number_of_samples(self, target_set=None):
        """ Return number of examples in target_set, if target_set is
        None return all examples.
        """
        if target_set == NNData.Set.TRAIN:
            return len(self._train_indices)
        if target_set == NNData.Set.TEST:
            return len(self._test_indices)
        if target_set is None:
            return len(self._features)

    def pool_is_empty(self, target_set=None):
        """ Return True if target_set pool is empty, False if not. """
        if target_set == NNData.Set.TRAIN or target_set is None:
            if len(self._train_pool) > 0:
                return False
            elif len(self._train_pool) == 0:
                return True
        if target_set == NNData.Set.TEST:
            if len(self._test_pool) > 0:
                return False
            elif len(self._test_pool) == 0:
                return True


class LayerType(Enum):
    """ Indicate which neural network layer neurode objects belong to. """
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class MultiLinkNode(ABC):
    """ Base class for all neurodes in neural network. """

    class Side(Enum):
        """ Indicate relationship between neurodes. """
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        self._reporting_nodes = {side: 0 for side in MultiLinkNode.Side}
        self._reference_value = {side: 0 for side in MultiLinkNode.Side}
        self._neighbors = {side: [] for side in MultiLinkNode.Side}

    def __str__(self):
        """ Change class representation to IDs of neurode objects. """
        main_node = id(self)
        upstream_nodes = [id(node) for node
                          in self._neighbors[MultiLinkNode.Side.UPSTREAM]]
        downstream_nodes = [id(node) for node
                            in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]]
        return f"This Node:\n[{main_node}] \n" \
               f"Nodes Upstream:\n{upstream_nodes}\n" \
               f"Nodes Downstream:\n{downstream_nodes}"

    @abstractmethod
    def _process_new_neighbor(self, node, side):
        """ Override in Neurode child class. """
        pass

    def reset_neighbors(self, nodes, side):
        """ Set node neighbors upstream or downstream and calculate
        reference value for node check-in.
        """
        self._neighbors[side] = list(nodes)
        for node in self._neighbors[side]:
            self._process_new_neighbor(node, side)
        self._reference_value[side] = (1 << len(self._neighbors[side])) - 1


class Neurode(MultiLinkNode):
    """ Implement MultiLinkNode parent class. """

    def __init__(self, node_type, learning_rate=.05):
        super().__init__()
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}

    def _process_new_neighbor(self, node, side):
        """ Generate random weights for upstream neighbors. """
        if side == Neurode.Side.UPSTREAM:
            self._weights[node] = random()

    def _check_in(self, node, side):
        """ Check if neighboring nodes have reported and return True if
        all nodes on given side have reported.
        """
        index = self._neighbors[side].index(node)
        self._reporting_nodes[side] |= 1 << index
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        else:
            return False

    def get_weight(self, node):
        """ Return weight of incoming upstream neurode connections. """
        return self._weights[node]

    @property
    def value(self):
        """ Return current value of neurode. """
        return self._value

    @property
    def node_type(self):
        """ Return role/layer of neurode. """
        return self._node_type

    @property
    def learning_rate(self):
        """ Return learning rate. """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        """ Set learning rate used in backpropagation. """
        self._learning_rate = learning_rate


class FFNeurode(Neurode):
    """ FeedForward neurode class, child of Neurode. """

    def __init__(self, node_type):
        super().__init__(node_type)

    @staticmethod
    def _sigmoid(value):
        """ Return result of sigmoid function at value. """
        return 1 / (1 + np.exp(-value))

    def _calculate_value(self):
        """ Calculate weighted sum of upstream nodes' values. """
        weighted_sum = 0
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            weighted_sum += node.value * self.get_weight(node)
        self._value = self._sigmoid(weighted_sum)

    def _fire_downstream(self):
        """ Communicate neurode-specific value to nodes downstream. """
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        """ Check in upstream neurodes. """
        checked_in = self._check_in(node, MultiLinkNode.Side.UPSTREAM)
        if checked_in:
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value):
        """ Set value for input layer neurodes. """
        if self._node_type == LayerType.INPUT:
            self._value = input_value
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)


class BPNeurode(Neurode):
    """ BackPropagation neurode class, child of Neurode. """

    def __init__(self, node_type):
        super().__init__(node_type)
        self._delta = 0

    @staticmethod
    def _sigmoid_derivative(value):
        """ Return derivative of sigmoid function at value. """
        return value * (1 - value)

    def _calculate_delta(self, expected_value=None):
        """ Calculate delta for neurode, based on node's layer. """
        if self._node_type == LayerType.OUTPUT:
            self._delta = (expected_value - self._value) \
                          * self._sigmoid_derivative(self._value)
        else:
            self._delta = 0
            for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                self._delta += (node.get_weight(self) * node.delta) \
                               * self._sigmoid_derivative(self._value)

    def data_ready_downstream(self, node):
        """ Check in downstream neurodes. """
        checked_in = self._check_in(node, MultiLinkNode.Side.DOWNSTREAM)
        if checked_in:
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value):
        """ Set expected value for output layer neurodes."""
        self._calculate_delta(expected_value)
        self._fire_upstream()

    def adjust_weights(self, node, adjustment):
        """ Add adjustment to upstream neurodes' weights. """
        self._weights[node] += adjustment

    def _update_weights(self):
        """ Update weights for all downstream neighbors. """
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adjustment = self._value * node.delta * node.learning_rate
            node.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        """ Communicate neurode-specific delta value to nodes upstream. """
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(self)

    @property
    def delta(self):
        """ Return delta. """
        return self._delta


class FFBPNeurode(FFNeurode, BPNeurode):
    """ FeedForward-BackPropagation Neurode, inherits from both
    FFNeurode and BPNeurode.
    """
    pass


class Node:
    """ Node class for DoublyLinkedList. """

    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    """ Implementation of doubly linked list ADT. """

    class EmptyListError(Exception):
        pass

    def __init__(self):
        self._head = None
        self._tail = None
        self._current = None

    def __iter__(self):
        self._current_iter = self._head
        return self

    def __next__(self):
        if self._current_iter is None:
            raise StopIteration
        ret_val = self._current_iter.data
        self._current_iter = self._current_iter.next
        return ret_val

    def move_forward(self):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        if self._current.next:
            self._current = self._current.next
        else:
            raise IndexError

    def move_back(self):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        if self._current.prev:
            self._current = self._current.prev
        else:
            raise IndexError

    def reset_to_head(self):
        if not self._head:
            raise DoublyLinkedList.EmptyListError
        self._current = self._head

    def reset_to_tail(self):
        if not self._tail:
            raise DoublyLinkedList.EmptyListError
        self._current = self._tail

    def add_to_head(self, data):
        new_node = Node(data)
        new_node.next = self._head
        if self._head:
            self._head.prev = new_node
        self._head = new_node
        if self._tail is None:
            self._tail = new_node
        self.reset_to_head()

    def add_after_curr(self, data):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        new_node = Node(data)
        new_node.prev = self._current
        new_node.next = self._current.next
        if self._current.next:
            self._current.next.prev = new_node
        self._current.next = new_node
        if self._tail == self._current:
            self._tail = new_node

    def remove_from_head(self):
        if not self._head:
            raise DoublyLinkedList.EmptyListError
        ret_val = self._head.data
        self._head = self._head.next
        if self._head:
            self._head.prev = None
        else:
            self._tail = None
        self.reset_to_head()
        return ret_val

    def remove_after_curr(self):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        if self._current == self._tail:
            raise IndexError
        ret_val = self._current.next.data
        if self._current.next == self._tail:
            self._tail = self._current
            self._current.next = None
        else:
            self._current.next = self._current.next.next
            self._current.next.prev = self._current
        return ret_val

    def get_current_data(self):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        return self._current.data


class LayerList(DoublyLinkedList):
    """ Doubly linked list of neural network layers. """

    def _link_with_next(self):
        """ Link node with upstream and downstream neighbors. """
        for node in self._current.data:
            node.reset_neighbors(self._current.next.data, FFBPNeurode.Side.DOWNSTREAM)
        for node in self._current.next.data:
            node.reset_neighbors(self._current.data, FFBPNeurode.Side.UPSTREAM)

    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        if inputs < 1 or outputs < 1:
            raise ValueError
        input_layer = [FFBPNeurode(LayerType.INPUT) for _ in range(inputs)]
        output_layer = [FFBPNeurode(LayerType.OUTPUT) for _ in range(outputs)]
        self.add_to_head(input_layer)
        self.add_after_curr(output_layer)
        self._link_with_next()

    def add_layer(self, num_nodes: int):
        """ Add hidden layer with specified number (num_nodes) of neurodes. """
        if self._current == self._tail:
            raise IndexError
        hidden_layer = [FFBPNeurode(LayerType.HIDDEN) for _ in range(num_nodes)]
        self.add_after_curr(hidden_layer)
        self._link_with_next()
        self.move_forward()
        self._link_with_next()
        self.move_back()

    def remove_layer(self):
        """ Remove layer after current layer from list. """
        if self._current == self._tail or self._current.next == self._tail:
            raise IndexError
        self.remove_after_curr()
        self._link_with_next()

    @property
    def input_nodes(self):
        return self._head.data

    @property
    def output_nodes(self):
        return self._tail.data


class FFBPNetwork:
    """ Create neural network layers with methods for training and testing. """

    class EmptySetException(Exception):
        pass

    def __init__(self, num_inputs: int, num_outputs: int):
        self._num_input_nodes = num_inputs
        self._num_output_nodes = num_outputs
        self._layer_list = LayerList(num_inputs, num_outputs)

    def add_hidden_layer(self, num_nodes: int, position=0):
        """ Add hidden layer to network at designated position. """
        self._layer_list.reset_to_head()
        for position in range(position):
            self._layer_list.move_forward()
        self._layer_list.add_layer(num_nodes)

    def train(self, data_set: NNData, epochs=1000, verbosity=2, order=NNData.Order.RANDOM):
        """ Train neural network on training set, and compute RMSE at intervals. """
        if data_set.number_of_samples(data_set.Set.TRAIN) == 0:
            raise FFBPNetwork.EmptySetException
        print("---TRAINING---")
        rmse_list = []
        for epoch in range(epochs):
            data_set.prime_data(order=order)
            squared_error = 0
            while not data_set.pool_is_empty(data_set.Set.TRAIN):
                feature, label = data_set.get_one_item(data_set.Set.TRAIN)
                nodes_and_inputs = list(zip(self._layer_list.input_nodes, feature))
                nodes_and_outputs = list(zip(self._layer_list.output_nodes, label))
                for input_node, input_value in nodes_and_inputs:
                    input_node.set_input(input_value)
                produced = []
                for output_node, expected_value in nodes_and_outputs:
                    output_node.set_expected(expected_value)
                    squared_error += (expected_value - output_node.value) ** 2 \
                        / self._num_output_nodes
                    produced.append(output_node.value)
                if epoch % 1000 == 0 and verbosity > 1:
                    print(f"Input: {feature} Expected: {label} Produced: {produced}")
                    print("---------")
            if epoch % 100 == 0 and verbosity > 0:
                rmse = np.sqrt(squared_error / data_set.number_of_samples(data_set.Set.TRAIN))
                print(f"Epoch {epoch} RMSE = {rmse}")
                rmse_list.append([epoch, rmse])
        print(f"Final RMSE = {rmse}")

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        """ Test neural network on testing set, and compute RMSE at the end. """
        if data_set.number_of_samples(data_set.Set.TEST) == 0:
            raise FFBPNetwork.EmptySetException
        plot_output = []
        print("---TESTING---")
        data_set.prime_data(data_set.Set.TEST, order)
        while not data_set.pool_is_empty(data_set.Set.TEST):
            feature, label = data_set.get_one_item(data_set.Set.TEST)
            nodes_and_inputs = list(zip(self._layer_list.input_nodes, feature))
            nodes_and_outputs = list(zip(self._layer_list.output_nodes, label))
            for input_node, input_value in nodes_and_inputs:
                input_node.set_input(input_value)
            for output_node, expected_value in nodes_and_outputs:
                output_node.set_expected(expected_value)
            print(f"Input values: {feature}")
            print(f"Expected output: {label}")
            print(f"Actual output:", end=" ")
            for node, label in nodes_and_outputs:
                print(f"{node.value}", end=" ")
                plot_output.append([float(feature), node.value, 'FFBPNetwork'])
            print("\n---------")
        squared_error = 0
        for output_node, label in nodes_and_outputs:
            squared_error += (label - output_node.value) ** 2 / self._num_output_nodes
        test_rmse = np.sqrt(squared_error / data_set.number_of_samples(data_set.Set.TEST))
        print(f"\nRMSE = {test_rmse}")
        return plot_output


def run_iris():
    """ Set up and run neural network with iris dataset examples. """
    network = FFBPNetwork(4, 3)
    network.add_hidden_layer(3)
    iris_x = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
              [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4],
              [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2],
              [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
              [4.8, 3.0, 1.4, 0.1], [4.3, 3.0, 1.1, 0.1], [5.8, 4.0, 1.2, 0.2],
              [5.7, 4.4, 1.5, 0.4], [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3],
              [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3], [5.4, 3.4, 1.7, 0.2],
              [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1.0, 0.2], [5.1, 3.3, 1.7, 0.5],
              [4.8, 3.4, 1.9, 0.2], [5.0, 3.0, 1.6, 0.2], [5.0, 3.4, 1.6, 0.4],
              [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2],
              [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [5.2, 4.1, 1.5, 0.1],
              [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.0, 3.2, 1.2, 0.2],
              [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3.0, 1.3, 0.2],
              [5.1, 3.4, 1.5, 0.2], [5.0, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3],
              [4.4, 3.2, 1.3, 0.2], [5.0, 3.5, 1.6, 0.6], [5.1, 3.8, 1.9, 0.4],
              [4.8, 3.0, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2],
              [5.3, 3.7, 1.5, 0.2], [5.0, 3.3, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4],
              [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4.0, 1.3],
              [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6],
              [4.9, 2.4, 3.3, 1.0], [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4],
              [5.0, 2.0, 3.5, 1.0], [5.9, 3.0, 4.2, 1.5], [6.0, 2.2, 4.0, 1.0],
              [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
              [5.6, 3.0, 4.5, 1.5], [5.8, 2.7, 4.1, 1.0], [6.2, 2.2, 4.5, 1.5],
              [5.6, 2.5, 3.9, 1.1], [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4.0, 1.3],
              [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2], [6.4, 2.9, 4.3, 1.3],
              [6.6, 3.0, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3.0, 5.0, 1.7],
              [6.0, 2.9, 4.5, 1.5], [5.7, 2.6, 3.5, 1.0], [5.5, 2.4, 3.8, 1.1],
              [5.5, 2.4, 3.7, 1.0], [5.8, 2.7, 3.9, 1.2], [6.0, 2.7, 5.1, 1.6],
              [5.4, 3.0, 4.5, 1.5], [6.0, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5],
              [6.3, 2.3, 4.4, 1.3], [5.6, 3.0, 4.1, 1.3], [5.5, 2.5, 4.0, 1.3],
              [5.5, 2.6, 4.4, 1.2], [6.1, 3.0, 4.6, 1.4], [5.8, 2.6, 4.0, 1.2],
              [5.0, 2.3, 3.3, 1.0], [5.6, 2.7, 4.2, 1.3], [5.7, 3.0, 4.2, 1.2],
              [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3.0, 1.1],
              [5.7, 2.8, 4.1, 1.3], [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9],
              [7.1, 3.0, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8], [6.5, 3.0, 5.8, 2.2],
              [7.6, 3.0, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8],
              [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2.0],
              [6.4, 2.7, 5.3, 1.9], [6.8, 3.0, 5.5, 2.1], [5.7, 2.5, 5.0, 2.0],
              [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3], [6.5, 3.0, 5.5, 1.8],
              [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6.0, 2.2, 5.0, 1.5],
              [6.9, 3.2, 5.7, 2.3], [5.6, 2.8, 4.9, 2.0], [7.7, 2.8, 6.7, 2.0],
              [6.3, 2.7, 4.9, 1.8], [6.7, 3.3, 5.7, 2.1], [7.2, 3.2, 6.0, 1.8],
              [6.2, 2.8, 4.8, 1.8], [6.1, 3.0, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1],
              [7.2, 3.0, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2.0],
              [6.4, 2.8, 5.6, 2.2], [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4],
              [7.7, 3.0, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4], [6.4, 3.1, 5.5, 1.8],
              [6.0, 3.0, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1], [6.7, 3.1, 5.6, 2.4],
              [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3],
              [6.7, 3.3, 5.7, 2.5], [6.7, 3.0, 5.2, 2.3], [6.3, 2.5, 5.0, 1.9],
              [6.5, 3.0, 5.2, 2.0], [6.2, 3.4, 5.4, 2.3], [5.9, 3.0, 5.1, 1.8]]
    iris_y = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
              [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
              [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
              [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
              [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
              [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
              [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
              [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
              [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
              [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
              [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
              [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
              [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
              [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
              [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
              [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
              [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
              [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
              [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    data = NNData(iris_x, iris_y, .7)
    network.train(data, epochs=10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_sin():
    """ Set up and run neural network with sin function examples. """
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    sin_x = [[0.00], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07],
             [0.08], [0.09], [0.10], [0.11], [0.12], [0.13], [0.14], [0.15],
             [0.16], [0.17], [0.18], [0.19], [0.20], [0.21], [0.22], [0.23],
             [0.24], [0.25], [0.26], [0.27], [0.28], [0.29], [0.30], [0.31],
             [0.32], [0.33], [0.34], [0.35], [0.36], [0.37], [0.38], [0.39],
             [0.40], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46], [0.47],
             [0.48], [0.49], [0.50], [0.51], [0.52], [0.53], [0.54], [0.55],
             [0.56], [0.57], [0.58], [0.59], [0.60], [0.61], [0.62], [0.63],
             [0.64], [0.65], [0.66], [0.67], [0.68], [0.69], [0.70], [0.71],
             [0.72], [0.73], [0.74], [0.75], [0.76], [0.77], [0.78], [0.79],
             [0.80], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86], [0.87],
             [0.88], [0.89], [0.90], [0.91], [0.92], [0.93], [0.94], [0.95],
             [0.96], [0.97], [0.98], [0.99], [1.00], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.10], [1.11],
             [1.12], [1.13], [1.14], [1.15], [1.16], [1.17], [1.18], [1.19],
             [1.20], [1.21], [1.22], [1.23], [1.24], [1.25], [1.26], [1.27],
             [1.28], [1.29], [1.30], [1.31], [1.32], [1.33], [1.34], [1.35],
             [1.36], [1.37], [1.38], [1.39], [1.40], [1.41], [1.42], [1.43],
             [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.50], [1.51],
             [1.52], [1.53], [1.54], [1.55], [1.56], [1.57]]
    sin_y = [[0], [0.00999983333416666], [0.0199986666933331], [0.0299955002024957],
             [0.0399893341866342], [0.0499791692706783], [0.0599640064794446],
             [0.0699428473375328], [0.0799146939691727], [0.089878549198011],
             [0.0998334166468282], [0.109778300837175], [0.119712207288919],
             [0.129634142619695], [0.139543114644236], [0.149438132473599],
             [0.159318206614246], [0.169182349066996], [0.179029573425824],
             [0.188858894976501], [0.198669330795061], [0.2084598998461],
             [0.218229623080869], [0.227977523535188], [0.237702626427135],
             [0.247403959254523], [0.257080551892155], [0.266731436688831],
             [0.276355648564114], [0.285952225104836], [0.29552020666134],
             [0.305058636443443], [0.314566560616118], [0.324043028394868],
             [0.333487092140814], [0.342897807455451], [0.35227423327509],
             [0.361615431964962], [0.370920469412983], [0.380188415123161],
             [0.389418342308651], [0.398609327984423], [0.40776045305957],
             [0.416870802429211], [0.425939465066], [0.43496553411123],
             [0.44394810696552], [0.452886285379068], [0.461779175541483],
             [0.470625888171158], [0.479425538604203], [0.488177246882907],
             [0.496880137843737], [0.505533341204847], [0.514135991653113],
             [0.522687228930659], [0.531186197920883], [0.539632048733969],
             [0.548023936791874], [0.556361022912784], [0.564642473395035],
             [0.572867460100481], [0.581035160537305], [0.58914475794227],
             [0.597195441362392], [0.60518640573604], [0.613116851973434],
             [0.62098598703656], [0.628793024018469], [0.636537182221968],
             [0.644217687237691], [0.651833771021537], [0.659384671971473],
             [0.666869635003698], [0.674287911628145], [0.681638760023334],
             [0.688921445110551], [0.696135238627357], [0.70327941920041],
             [0.710353272417608], [0.717356090899523], [0.724287174370143],
             [0.731145829726896], [0.737931371109963], [0.744643119970859],
             [0.751280405140293], [0.757842562895277], [0.764328937025505],
             [0.770738878898969], [0.777071747526824], [0.783326909627483],
             [0.78950373968995], [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998],
             [0.82488571333845], [0.83049737049197], [0.836025978600521],
             [0.841470984807897], [0.846831844618015], [0.852108021949363],
             [0.857298989188603], [0.862404227243338], [0.867423225594017],
             [0.872355482344986], [0.877200504274682], [0.881957806884948],
             [0.886626914449487], [0.891207360061435], [0.895698685680048],
             [0.900100442176505], [0.904412189378826], [0.908633496115883],
             [0.912763940260521], [0.916803108771767], [0.920750597736136],
             [0.92460601240802], [0.928368967249167], [0.932039085967226],
             [0.935616001553386], [0.939099356319068], [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516],
             [0.955100855584692], [0.958015860289225], [0.960835064206073],
             [0.963558185417193], [0.966184951612734], [0.968715100118265],
             [0.971148377921045], [0.973484541695319], [0.975723357826659],
             [0.977864602435316], [0.979908061398614], [0.98185353037236],
             [0.983700814811277], [0.98544972998846], [0.98710010101385],
             [0.98865176285172], [0.990104560337178], [0.991458348191686],
             [0.992712991037588], [0.993868363411645], [0.994924349777581],
             [0.99588084453764], [0.996737752043143], [0.997494986604054],
             [0.998152472497548], [0.998710143975583], [0.999167945271476],
             [0.999525830605479], [0.999783764189357], [0.999941720229966],
             [0.999999682931835]]
    data = NNData(sin_x, sin_y, .7)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    sin_output = network.test(data)

    # For generating plot
    x_coord = [item[0] for item in sin_x]
    y_coord = [item[0] for item in sin_y]
    sin_coordinates = list(zip(x_coord, y_coord))
    sin_expected = []
    for x_value, y_value in sin_coordinates:
        sin_expected.append([x_value, y_value, 'sin(x)'])
    sin_master = sin_output + sin_expected
    df = pd.DataFrame(sin_master, columns=['x_value', 'y_value', 'type'])
    ax = sns.lineplot(data=df, x='x_value', y='y_value', hue='type')
    ax.set_title("Sin(x) vs neural network's approximation of sin(x)")
    ax.set_xlabel('Input (Radians)')
    ax.set_ylabel('Output')
    plt.show()


# noinspection PyPep8Naming
def run_XOR():
    """ Set up and run neural network with XOR examples. """
    XOR_X = [[0, 0], [1, 0], [0, 1], [1, 1]]
    XOR_Y = [[0], [1], [1], [0]]
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(3)
    data = NNData(XOR_X, XOR_Y, 1)
    network.train(data, 20001, order=NNData.Order.RANDOM)


if __name__ == "__main__":
    run_sin()

"""
---run_iris() sample run---
---TRAINING---
Input: [5.  3.6 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.847212041380112, 0.872915942896514, 0.852743102944124]
---------
Input: [7.2 3.6 6.1 2.5] Expected: [0. 0. 1.] Produced: [0.8567743788803999, 0.8789250087147014, 0.857395912730836]
---------
Input: [5.4 3.9 1.3 0.4] Expected: [1. 0. 0.] Produced: [0.8479307262559681, 0.871823441215176, 0.8528941029132412]
---------
Input: [6.9 3.1 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.8542878168760561, 0.8751616911492844, 0.855177579610208]
---------
Input: [5.7 2.5 5.  2. ] Expected: [0. 0. 1.] Produced: [0.8521199833857909, 0.875227015406552, 0.8530807338036468]
---------
Input: [5.1 3.3 1.7 0.5] Expected: [1. 0. 0.] Produced: [0.8435842062735023, 0.8683389699252777, 0.8488080547569334]
---------
Input: [4.9 2.5 4.5 1.7] Expected: [0. 0. 1.] Produced: [0.8492938292948917, 0.8711003152724501, 0.8506088488272734]
---------
Input: [5.1 3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.8397488056790918, 0.8633083004720925, 0.8456466001859939]
---------
Input: [5.  3.2 1.2 0.2] Expected: [1. 0. 0.] Produced: [0.8386727262896573, 0.8603512093419978, 0.8424692338831705]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.8381543646314559, 0.8578410087631552, 0.8397355755631729]
---------
Input: [6.  3.  4.8 1.8] Expected: [0. 0. 1.] Produced: [0.8494708915954265, 0.8652284270380936, 0.8455209733566386]
---------
Input: [7.  3.2 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.8474741527671398, 0.8635365341307584, 0.845961894069596]
---------
Input: [5.2 3.5 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.8366041247645537, 0.8565669813604944, 0.8376744150328743]
---------
Input: [6.8 2.8 4.8 1.4] Expected: [0. 1. 0.] Produced: [0.8455398254430516, 0.8618232955152322, 0.84147203281609]
---------
Input: [5.1 3.8 1.5 0.3] Expected: [1. 0. 0.] Produced: [0.8354993084109361, 0.8556097404021235, 0.8338023101373526]
---------
Input: [6.4 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.8431923484574027, 0.8597729139540324, 0.8365516487322074]
---------
Input: [6.8 3.  5.5 2.1] Expected: [0. 0. 1.] Produced: [0.8424744997377587, 0.8612748340902687, 0.8352307930848746]
---------
Input: [5.6 3.  4.1 1.3] Expected: [0. 1. 0.] Produced: [0.8379763531382849, 0.8575736206509255, 0.834162352882627]
---------
Input: [5.4 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.8360952647556652, 0.8582127695072016, 0.8320820267828026]
---------
Input: [4.8 3.  1.4 0.3] Expected: [1. 0. 0.] Produced: [0.823680950986569, 0.8501490107392802, 0.8225186822854005]
---------
Input: [5.3 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.8264726519937522, 0.8502317770811971, 0.8219186701201311]
---------
Input: [6.4 2.7 5.3 1.9] Expected: [0. 0. 1.] Produced: [0.8361076747193266, 0.8558150558639978, 0.8256600441645684]
---------
Input: [5.4 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.8246523644178427, 0.8463763192014565, 0.8200264853717039]
---------
Input: [6.9 3.1 5.1 2.3] Expected: [0. 0. 1.] Produced: [0.8348189341133984, 0.8523294629449136, 0.8240421356616969]
---------
Input: [6.1 2.8 4.7 1.2] Expected: [0. 1. 0.] Produced: [0.8303026870579142, 0.8485569366425105, 0.8231780253386328]
---------
Input: [4.6 3.1 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.8159098732869862, 0.8389484670663869, 0.8122940000915269]
---------
Input: [6.  2.2 5.  1.5] Expected: [0. 0. 1.] Produced: [0.8287306435726681, 0.8471047506746521, 0.8181951765464774]
---------
Input: [6.7 3.3 5.7 2.1] Expected: [0. 0. 1.] Produced: [0.8279298757453012, 0.8463774483413194, 0.8199098900986567]
---------
Input: [5.5 3.5 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.8147223140547528, 0.8354036438515834, 0.8133033409391032]
---------
Input: [6.5 2.8 4.6 1.5] Expected: [0. 1. 0.] Produced: [0.824649257130889, 0.8409480661934093, 0.8169803667821034]
---------
Input: [6.7 3.1 4.7 1.5] Expected: [0. 1. 0.] Produced: [0.8223375224341563, 0.8415731439247917, 0.8144400407285387]
---------
Input: [5.1 3.8 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.8094775538494484, 0.8335716693658618, 0.8049546264875715]
---------
Input: [6.3 3.4 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.8214491656149051, 0.8406715666597235, 0.809603309636254]
---------
Input: [5.6 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.8165541250403038, 0.8365778300402846, 0.8088343798522144]
---------
Input: [6.5 3.  5.2 2. ] Expected: [0. 0. 1.] Produced: [0.815589343623396, 0.8384412725948003, 0.8070881614100684]
---------
Input: [5.8 2.7 3.9 1.2] Expected: [0. 1. 0.] Produced: [0.8099094774387195, 0.8337536734044718, 0.8058861466680077]
---------
Input: [6.1 3.  4.9 1.8] Expected: [0. 0. 1.] Produced: [0.8093598831776045, 0.836067020054944, 0.8044666021589895]
---------
Input: [6.5 3.2 5.1 2. ] Expected: [0. 0. 1.] Produced: [0.8071137175713773, 0.834233089742385, 0.8055700094967332]
---------
Input: [5.  2.  3.5 1. ] Expected: [0. 1. 0.] Produced: [0.7979787703603723, 0.8267165250284247, 0.8021340098015916]
---------
Input: [7.9 3.8 6.4 2. ] Expected: [0. 0. 1.] Produced: [0.8020888624330582, 0.833044490081519, 0.803899487104801]
---------
Input: [5.5 2.4 3.7 1. ] Expected: [0. 1. 0.] Produced: [0.7934122747121342, 0.8260283864681548, 0.8009987687647354]
---------
Input: [5.7 3.8 1.7 0.3] Expected: [1. 0. 0.] Produced: [0.7857967124971984, 0.822901024575832, 0.7952698964616725]
---------
Input: [5.5 2.4 3.8 1.1] Expected: [0. 1. 0.] Produced: [0.7914691870838191, 0.8243375271891524, 0.7952779070887209]
---------
Input: [7.2 3.2 6.  1.8] Expected: [0. 0. 1.] Produced: [0.7929765024522458, 0.8286411407845954, 0.7951645812792324]
---------
Input: [4.8 3.4 1.9 0.2] Expected: [1. 0. 0.] Produced: [0.7756946641969311, 0.8146177056010997, 0.7870314657916377]
---------
Input: [6.6 2.9 4.6 1.3] Expected: [0. 1. 0.] Produced: [0.7887202298110079, 0.8220198907708877, 0.7916893812608372]
---------
Input: [4.7 3.2 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.7707149720901542, 0.8103544474787956, 0.7789963470047548]
---------
Input: [6.4 3.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.7866451524273748, 0.8201862893921362, 0.7855013004086202]
---------
Input: [6.3 2.7 4.9 1.8] Expected: [0. 0. 1.] Produced: [0.7839008415294292, 0.8211801002649853, 0.7825137416023976]
---------
Input: [5.6 2.9 3.6 1.3] Expected: [0. 1. 0.] Produced: [0.7776920949892283, 0.8162254972509702, 0.7816879687222235]
---------
Input: [6.6 3.  4.4 1.4] Expected: [0. 1. 0.] Produced: [0.7763112874894034, 0.8184539749585378, 0.7796011118396109]
---------
Input: [6.3 2.8 5.1 1.5] Expected: [0. 0. 1.] Produced: [0.7730341442530095, 0.8191804275248916, 0.7763158506927397]
---------
Input: [4.9 3.  1.4 0.2] Expected: [1. 0. 0.] Produced: [0.7542810996683702, 0.8038772604603548, 0.7676419632476404]
---------
Input: [4.7 3.2 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.7551490209942224, 0.8010372874734074, 0.7643487255237695]
---------
Input: [6.2 2.8 4.8 1.8] Expected: [0. 0. 1.] Produced: [0.7719884365070788, 0.8113175203336708, 0.7707174953331438]
---------
Input: [6.2 2.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.7671919519708676, 0.8074711888581223, 0.770985823511707]
---------
Input: [5.1 3.5 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.7502935778555184, 0.7971984320668308, 0.7594441206951089]
---------
Input: [5.1 2.5 3.  1.1] Expected: [0. 1. 0.] Produced: [0.7598654987335867, 0.8012884916370001, 0.7610182656738987]
---------
Input: [6.4 3.1 5.5 1.8] Expected: [0. 0. 1.] Produced: [0.7626924901643986, 0.80729159774623, 0.7613012144073793]
---------
Input: [6.  3.4 4.5 1.6] Expected: [0. 1. 0.] Produced: [0.7577056777740927, 0.8033603916694106, 0.7617260054767753]
---------
Input: [5.  2.3 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.7477359053862142, 0.7989135674618044, 0.7542705280320612]
---------
Input: [6.9 3.2 5.7 2.3] Expected: [0. 0. 1.] Produced: [0.7525931788720605, 0.8067410410069643, 0.7556944749128359]
---------
Input: [4.9 2.4 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.7396973963529767, 0.7963955702532574, 0.7516152789853356]
---------
Input: [5.9 3.  5.1 1.8] Expected: [0. 0. 1.] Produced: [0.7427136417747465, 0.8028635509125464, 0.7520081835973743]
---------
Input: [5.8 2.7 4.1 1. ] Expected: [0. 1. 0.] Produced: [0.7345632761323494, 0.7964936715042502, 0.7508688690125744]
---------
Input: [6.7 2.5 5.8 1.8] Expected: [0. 0. 1.] Produced: [0.7353964900537572, 0.8012204938599138, 0.7498537307045966]
---------
Input: [4.3 3.  1.1 0.1] Expected: [1. 0. 0.] Produced: [0.7079575688116866, 0.7782721887536987, 0.7368213333297507]
---------
Input: [7.4 2.8 6.1 1.9] Expected: [0. 0. 1.] Produced: [0.733615484763116, 0.7958420297174846, 0.7478937326850781]
---------
Input: [5.9 3.  4.2 1.5] Expected: [0. 1. 0.] Produced: [0.7260669912644362, 0.7899506161150511, 0.7473078980310801]
---------
Input: [4.8 3.1 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.7057604383626755, 0.7770680256866501, 0.7338395710448399]
---------
Input: [5.6 2.5 3.9 1.1] Expected: [0. 1. 0.] Produced: [0.7203002867676248, 0.784976774909569, 0.7377816207034031]
---------
Input: [7.3 2.9 6.3 1.8] Expected: [0. 0. 1.] Produced: [0.7222730886948541, 0.790942319603432, 0.7372873951786351]
---------
Input: [4.4 2.9 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.6955516517517213, 0.7684663680858141, 0.7253905292905661]
---------
Input: [6.5 3.  5.5 1.8] Expected: [0. 0. 1.] Produced: [0.7186285850651992, 0.7836735600101415, 0.7342935343343943]
---------
Input: [6.3 3.3 6.  2.5] Expected: [0. 0. 1.] Produced: [0.715939042285923, 0.7816964236856064, 0.7366505340035806]
---------
Input: [6.3 2.5 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.7080510490276759, 0.7754228301863854, 0.7361660333158989]
---------
Input: [5.4 3.4 1.7 0.2] Expected: [1. 0. 0.] Produced: [0.689103800223777, 0.7640287734814736, 0.7238485071169859]
---------
Input: [5.2 2.7 3.9 1.4] Expected: [0. 1. 0.] Produced: [0.7024759010934888, 0.770487705540408, 0.7264876135581041]
---------
Input: [7.7 2.6 6.9 2.3] Expected: [0. 0. 1.] Produced: [0.7049266291900114, 0.7772155546467825, 0.7261004702269134]
---------
Input: [5.  3.3 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.6791124617549542, 0.7555093957336015, 0.7156445277194092]
---------
Input: [6.4 2.8 5.6 2.1] Expected: [0. 0. 1.] Produced: [0.7006033751020394, 0.7688585876566844, 0.7227370442257274]
---------
Input: [6.2 3.4 5.4 2.3] Expected: [0. 0. 1.] Produced: [0.6964184371607556, 0.7655771433449363, 0.7245513324848067]
---------
Input: [4.5 2.3 1.3 0.3] Expected: [1. 0. 0.] Produced: [0.6685818186333443, 0.7413222452269962, 0.7118577019083033]
---------
Input: [7.7 3.8 6.7 2.2] Expected: [0. 0. 1.] Produced: [0.6949775641489094, 0.7593000531019672, 0.7227657510777515]
---------
Input: [5.  3.  1.6 0.2] Expected: [1. 0. 0.] Produced: [0.6682193543563815, 0.7364971917280906, 0.7116752638687152]
---------
Input: [6.7 3.1 4.4 1.4] Expected: [0. 1. 0.] Produced: [0.6877589249227498, 0.7479581619894948, 0.7178567267984022]
---------
Input: [6.7 3.3 5.7 2.5] Expected: [0. 0. 1.] Produced: [0.6873373614273867, 0.752703946807197, 0.7158052634888183]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.6585573933302687, 0.7279772195001405, 0.7038021613579699]
---------
Input: [5.8 2.7 5.1 1.9] Expected: [0. 0. 1.] Produced: [0.6816555305366315, 0.7424383083751431, 0.7117589561512313]
---------
Input: [5.  3.5 1.6 0.6] Expected: [1. 0. 0.] Produced: [0.662893614219204, 0.7264550637891037, 0.7056792207320062]
---------
Input: [6.7 3.1 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.6817913091409594, 0.7367456378694478, 0.7106870670271694]
---------
Input: [5.2 3.4 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.6551615642743176, 0.7140295430750231, 0.7002962670661105]
---------
Input: [5.7 2.8 4.5 1.3] Expected: [0. 1. 0.] Produced: [0.6724465191952683, 0.7229909920743154, 0.704696015120929]
---------
Input: [6.4 3.2 5.3 2.3] Expected: [0. 0. 1.] Produced: [0.673773509015328, 0.7295132165587829, 0.7034932727192915]
---------
Input: [4.6 3.2 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.6447441399429947, 0.7044502328843267, 0.6917590831369174]
---------
Input: [6.1 2.9 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.6662817930806586, 0.7171437388045697, 0.6985720854367075]
---------
Input: [4.6 3.6 1.  0.2] Expected: [1. 0. 0.] Produced: [0.641514023528249, 0.7013770076882502, 0.6832107670740548]
---------
Input: [5.8 2.8 5.1 2.4] Expected: [0. 0. 1.] Produced: [0.6682905565096557, 0.7183760283429688, 0.6921594993849686]
---------
Input: [7.7 3.  6.1 2.3] Expected: [0. 0. 1.] Produced: [0.6649404925573842, 0.7152896281380847, 0.6949079791368312]
---------
Input: [5.7 4.4 1.5 0.4] Expected: [1. 0. 0.] Produced: [0.6412786325793153, 0.6951543064496921, 0.6870848463903662]
---------
Input: [5.  3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.637589526890613, 0.6856917562661629, 0.6793086891517917]
---------
Input: [5.  3.4 1.6 0.4] Expected: [1. 0. 0.] Produced: [0.6424980401660773, 0.6838089508267563, 0.6766319833717078]
---------
Input: [6.2 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.6593501868081127, 0.6916935182980867, 0.6799632192245874]
---------
Input: [5.6 2.7 4.2 1.3] Expected: [0. 1. 0.] Produced: [0.6532228530979349, 0.6923891466979323, 0.6747241741644429]
---------
Input: [5.9 3.2 4.8 1.8] Expected: [0. 1. 0.] Produced: [0.6527157854234101, 0.697754701306165, 0.6722584023177305]
---------
Epoch 0 RMSE = 0.6519055706850148
Epoch 100 RMSE = 0.32636538197830717
Epoch 200 RMSE = 0.18151305483769423
Epoch 300 RMSE = 0.1570933084799025
Epoch 400 RMSE = 0.14530591155869854
Epoch 500 RMSE = 0.13142217469453554
Epoch 600 RMSE = 0.15757887778341356
Epoch 700 RMSE = 0.14990434815891204
Epoch 800 RMSE = 0.12342266801975413
Epoch 900 RMSE = 0.11701307890392268
Input: [6.2 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.02331188082816719, 0.9713145105962505, 0.02655316230126943]
---------
Input: [5.1 3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9913197897209407, 0.010980621444991689, 0.009348188068492724]
---------
Input: [6.1 2.9 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.019275515588807957, 0.9537111331235802, 0.043821122596347746]
---------
Input: [7.2 3.2 6.  1.8] Expected: [0. 0. 1.] Produced: [0.003228325030087596, 0.09691075681473738, 0.9044940647603757]
---------
Input: [6.9 3.2 5.7 2.3] Expected: [0. 0. 1.] Produced: [0.0026694157341341234, 0.057762866719787254, 0.943500188475457]
---------
Input: [7.9 3.8 6.4 2. ] Expected: [0. 0. 1.] Produced: [0.00669342485870768, 0.47997333106472123, 0.5169799392491831]
---------
Input: [5.9 3.  5.1 1.8] Expected: [0. 0. 1.] Produced: [0.002676915243972275, 0.05720262144978759, 0.9440184806518944]
---------
Input: [5.1 3.3 1.7 0.5] Expected: [1. 0. 0.] Produced: [0.9790024656168106, 0.02519611959381896, 0.004924714847575236]
---------
Input: [5.  2.  3.5 1. ] Expected: [0. 1. 0.] Produced: [0.02028624256742546, 0.9519303834514502, 0.044316253821099036]
---------
Input: [4.8 3.4 1.9 0.2] Expected: [1. 0. 0.] Produced: [0.9812922231575378, 0.02260938201434651, 0.006211551480345869]
---------
Input: [4.3 3.  1.1 0.1] Expected: [1. 0. 0.] Produced: [0.9934319763916842, 0.008422773145583462, 0.012064180275666823]
---------
Input: [4.8 3.1 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9859342271502138, 0.017277446576539856, 0.007090724359990028]
---------
Input: [7.7 2.6 6.9 2.3] Expected: [0. 0. 1.] Produced: [0.0026523412614194043, 0.05635108823677628, 0.9449042309523269]
---------
Input: [5.7 4.4 1.5 0.4] Expected: [1. 0. 0.] Produced: [0.9945531935466274, 0.0070576145762152355, 0.013441628983986187]
---------
Input: [6.7 3.3 5.7 2.1] Expected: [0. 0. 1.] Produced: [0.00266058553172441, 0.05661847820647447, 0.9446255414756767]
---------
Input: [6.1 3.  4.9 1.8] Expected: [0. 0. 1.] Produced: [0.002842725971047249, 0.06709628722572251, 0.9341183338729061]
---------
Input: [5.9 3.  4.2 1.5] Expected: [0. 1. 0.] Produced: [0.020270167084541022, 0.9567256006664052, 0.040226006664819826]
---------
Input: [5.  3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9912362389114489, 0.011054665969043788, 0.00934243591689015]
---------
Input: [4.9 2.4 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.02592663776135006, 0.9667964136810601, 0.027804435361032106]
---------
Input: [7.4 2.8 6.1 1.9] Expected: [0. 0. 1.] Produced: [0.002660773529648993, 0.056719118566414935, 0.9445352317044643]
---------
Input: [5.2 3.5 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9920707983098916, 0.010058071567346942, 0.009999804400580552]
---------
Input: [5.8 2.7 5.1 1.9] Expected: [0. 0. 1.] Produced: [0.0026583574223630053, 0.05626683537402783, 0.9449651236770955]
---------
Input: [6.6 2.9 4.6 1.3] Expected: [0. 1. 0.] Produced: [0.022656621170930997, 0.9700274909071395, 0.028041828052803825]
---------
Input: [5.  3.3 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9917933896752603, 0.010389512545347064, 0.009816670314826238]
---------
Input: [5.1 3.5 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9929599312823907, 0.008990143602838927, 0.011011622393733959]
---------
Input: [6.7 3.1 4.4 1.4] Expected: [0. 1. 0.] Produced: [0.02338286335365949, 0.9711018891951508, 0.026594154590238055]
---------
Input: [5.4 3.9 1.3 0.4] Expected: [1. 0. 0.] Produced: [0.9940253822106601, 0.007700236098953074, 0.0124018543395402]
---------
Input: [6.9 3.1 5.1 2.3] Expected: [0. 0. 1.] Produced: [0.0027313361306301, 0.06061187067788783, 0.9406108088326737]
---------
Input: [7.7 3.8 6.7 2.2] Expected: [0. 0. 1.] Produced: [0.0026568909777786114, 0.05649396894852048, 0.9447564533730605]
---------
Input: [5.2 3.4 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.992543004823762, 0.00949070521459791, 0.010501775276862347]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9900626548489884, 0.01244486705747725, 0.008879615529771057]
---------
Input: [4.7 3.2 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9870580125723892, 0.015965593211599093, 0.007443937752138081]
---------
Input: [6.4 2.7 5.3 1.9] Expected: [0. 0. 1.] Produced: [0.002662537336712797, 0.0565585825711016, 0.9446773049371125]
---------
Input: [6.9 3.1 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.021438736941274884, 0.9658035832858113, 0.032198118107876636]
---------
Input: [5.6 2.7 4.2 1.3] Expected: [0. 1. 0.] Produced: [0.015312503191223566, 0.9070285925540089, 0.08834207266411508]
---------
Input: [5.  3.5 1.6 0.6] Expected: [1. 0. 0.] Produced: [0.9830883578184141, 0.020538070994832108, 0.0053623877551912825]
---------
Input: [5.7 2.5 5.  2. ] Expected: [0. 0. 1.] Produced: [0.0026563805595016363, 0.05619248166809009, 0.9450412553098744]
---------
Input: [6.1 2.8 4.7 1.2] Expected: [0. 1. 0.] Produced: [0.01727773327921, 0.9359670328634737, 0.060922388638621164]
---------
Input: [7.2 3.6 6.1 2.5] Expected: [0. 0. 1.] Produced: [0.0026536625126651033, 0.05627711647856883, 0.9449685821216337]
---------
Input: [5.4 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.0035966519036135423, 0.12213845756672785, 0.8785758196782568]
---------
Input: [6.7 3.1 4.7 1.5] Expected: [0. 1. 0.] Produced: [0.02239454437633951, 0.969538014348686, 0.028535967408811265]
---------
Input: [5.1 3.8 1.5 0.3] Expected: [1. 0. 0.] Produced: [0.9926259695376766, 0.009403979298113058, 0.010447188898121011]
---------
Input: [7.3 2.9 6.3 1.8] Expected: [0. 0. 1.] Produced: [0.002660514742075996, 0.057133531384139305, 0.944129920524054]
---------
Input: [6.  3.  4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0030242366835722683, 0.07974594332377309, 0.9214675252987109]
---------
Input: [4.9 2.5 4.5 1.7] Expected: [0. 0. 1.] Produced: [0.0026677939446678135, 0.05653832684895883, 0.944649381067811]
---------
Input: [5.  3.2 1.2 0.2] Expected: [1. 0. 0.] Produced: [0.9932120058158638, 0.008695339640410636, 0.011369858825844471]
---------
Input: [6.2 2.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.00504426394885971, 0.28039515293469836, 0.7189979013464513]
---------
Input: [5.1 3.8 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9925684929944742, 0.009491899642918702, 0.010481143418879641]
---------
Input: [6.4 3.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.023093660161343826, 0.9714209474393307, 0.02654010261431357]
---------
Input: [5.8 2.7 4.1 1. ] Expected: [0. 1. 0.] Produced: [0.023758298111745558, 0.9711293784307719, 0.026510987326029155]
---------
Input: [6.7 3.3 5.7 2.5] Expected: [0. 0. 1.] Produced: [0.002654948053188937, 0.05739388422016187, 0.9438706639152356]
---------
Input: [5.9 3.2 4.8 1.8] Expected: [0. 1. 0.] Produced: [0.004550547495286357, 0.2252714079765036, 0.7747227118473282]
---------
Input: [6.  3.4 4.5 1.6] Expected: [0. 1. 0.] Produced: [0.022898416966534062, 0.9709406930012313, 0.026991648689500655]
---------
Input: [6.4 3.2 5.3 2.3] Expected: [0. 0. 1.] Produced: [0.0027156559571471974, 0.061591628716726465, 0.9396924915521426]
---------
Input: [5.5 2.4 3.7 1. ] Expected: [0. 1. 0.] Produced: [0.02435643924759813, 0.9707204881948988, 0.02634258358202307]
---------
Input: [5.6 2.9 3.6 1.3] Expected: [0. 1. 0.] Produced: [0.02582937942146047, 0.9691135208955395, 0.025584000447920273]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9899894892477188, 0.012610777283128557, 0.008761952346745124]
---------
Input: [5.4 3.4 1.7 0.2] Expected: [1. 0. 0.] Produced: [0.988641926396616, 0.014212937501273434, 0.007604782603249265]
---------
Input: [6.5 3.2 5.1 2. ] Expected: [0. 0. 1.] Produced: [0.00656413846842561, 0.4674106100005691, 0.5294456398263259]
---------
Input: [5.5 3.5 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9939459527764649, 0.007819513932510746, 0.012359253521689371]
---------
Input: [5.4 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9931957758732988, 0.00873358688586672, 0.011199260239362593]
---------
Input: [5.3 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9931276356143136, 0.008816328918378868, 0.011126513146307289]
---------
Input: [6.3 2.5 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.0056019861779933525, 0.35180359470326095, 0.6468182179225074]
---------
Input: [5.6 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.017922851221085553, 0.9432573665088284, 0.05378531248826372]
---------
Input: [4.6 3.2 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9905523831781546, 0.011945024083705093, 0.00895652099435063]
---------
Input: [5.  3.4 1.6 0.4] Expected: [1. 0. 0.] Produced: [0.9861872333539153, 0.017117555558203722, 0.006436917790883839]
---------
Input: [6.8 3.  5.5 2.1] Expected: [0. 0. 1.] Produced: [0.0029801669086982267, 0.07971522235175212, 0.9216750591449545]
---------
Input: [4.9 3.  1.4 0.2] Expected: [1. 0. 0.] Produced: [0.989289304035739, 0.013451389774290156, 0.00815655614212844]
---------
Input: [5.  3.6 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9932996814395824, 0.008623946166204339, 0.011407894127906019]
---------
Input: [5.5 2.4 3.8 1.1] Expected: [0. 1. 0.] Produced: [0.0237912120919049, 0.9711529575235349, 0.02639768092650585]
---------
Input: [6.3 3.3 6.  2.5] Expected: [0. 0. 1.] Produced: [0.002652195670445026, 0.05806248753755635, 0.943225288257085]
---------
Input: [6.4 3.1 5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0033050512048826792, 0.10478860212490074, 0.8966116201218381]
---------
Input: [6.3 2.7 4.9 1.8] Expected: [0. 0. 1.] Produced: [0.005603197120050846, 0.35490781859385395, 0.6436724778543528]
---------
Input: [6.4 2.8 5.6 2.1] Expected: [0. 0. 1.] Produced: [0.0026561769191278967, 0.05778039217447248, 0.9434963408347955]
---------
Input: [6.3 3.4 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0026557093790280156, 0.05771806963043786, 0.943550928545338]
---------
Input: [6.  2.2 5.  1.5] Expected: [0. 0. 1.] Produced: [0.0028192877361405797, 0.06758468356096581, 0.9337391926510734]
---------
Input: [6.6 3.  4.4 1.4] Expected: [0. 1. 0.] Produced: [0.02330718654668393, 0.9718101280995411, 0.026085530142937047]
---------
Input: [4.4 2.9 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9870889870636401, 0.016021043206294616, 0.007610186524341933]
---------
Input: [5.8 2.8 5.1 2.4] Expected: [0. 0. 1.] Produced: [0.0026541351284195603, 0.05754795799906162, 0.9437150434046663]
---------
Input: [5.1 2.5 3.  1.1] Expected: [0. 1. 0.] Produced: [0.03227944110403485, 0.9617533165854866, 0.024498024217511877]
---------
Input: [5.  3.  1.6 0.2] Expected: [1. 0. 0.] Produced: [0.984922505588085, 0.01855073065601767, 0.006703577051033998]
---------
Input: [5.6 3.  4.1 1.3] Expected: [0. 1. 0.] Produced: [0.023470945514767874, 0.9706386485447769, 0.026885490456265582]
---------
Input: [6.5 2.8 4.6 1.5] Expected: [0. 1. 0.] Produced: [0.022355071648002964, 0.9699916118403134, 0.028181811363488858]
---------
Input: [5.2 2.7 3.9 1.4] Expected: [0. 1. 0.] Produced: [0.019944242542327868, 0.9543587599709441, 0.04239989636367933]
---------
Input: [6.7 3.1 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0026576255633570627, 0.05784843338620339, 0.9434250082371108]
---------
Input: [6.3 2.8 5.1 1.5] Expected: [0. 0. 1.] Produced: [0.007863500053848772, 0.5974159315601203, 0.39798898169761915]
---------
Input: [5.  2.3 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.025690872373463357, 0.9679227930069526, 0.02723268091752501]
---------
Input: [5.8 2.7 3.9 1.2] Expected: [0. 1. 0.] Produced: [0.02385678342234204, 0.9702785071145473, 0.026890950439782093]
---------
Input: [7.7 3.  6.1 2.3] Expected: [0. 0. 1.] Produced: [0.0026578105826688837, 0.05733967681858111, 0.9439305244762628]
---------
Input: [5.6 2.5 3.9 1.1] Expected: [0. 1. 0.] Produced: [0.023297814943413847, 0.9693459633019585, 0.028141502849120578]
---------
Input: [4.5 2.3 1.3 0.3] Expected: [1. 0. 0.] Produced: [0.975340642637759, 0.029418228127120805, 0.0057408316356860984]
---------
Input: [5.7 2.8 4.5 1.3] Expected: [0. 1. 0.] Produced: [0.011998753772913434, 0.83162288370239, 0.16282066487149793]
---------
Input: [4.6 3.1 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9878233051377133, 0.01511989452573744, 0.007728443851609265]
---------
Input: [4.7 3.2 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9919922390321163, 0.010175096506508314, 0.010073875162413202]
---------
Input: [6.8 2.8 4.8 1.4] Expected: [0. 1. 0.] Produced: [0.021929249506957755, 0.968319581181173, 0.029829166632447775]
---------
Input: [7.  3.2 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.023131303422729357, 0.9717719453401403, 0.026276680638293158]
---------
Input: [6.2 3.4 5.4 2.3] Expected: [0. 0. 1.] Produced: [0.002656342493012847, 0.05712522932072743, 0.9441264169886937]
---------
Input: [6.5 3.  5.2 2. ] Expected: [0. 0. 1.] Produced: [0.002734267402119183, 0.06169077449786739, 0.9395787793550912]
---------
Input: [6.5 3.  5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0027070068966143328, 0.06008778277759101, 0.9411911762431963]
---------
Input: [6.7 2.5 5.8 1.8] Expected: [0. 0. 1.] Produced: [0.0026544339458427837, 0.05709837575511411, 0.9441715186444414]
---------
Input: [4.6 3.6 1.  0.2] Expected: [1. 0. 0.] Produced: [0.9954098934316598, 0.006011538417174708, 0.015881212475695298]
---------
Input: [5.7 3.8 1.7 0.3] Expected: [1. 0. 0.] Produced: [0.9909885001644934, 0.011373134452812171, 0.008746878194650155]
---------
Input: [6.2 2.8 4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0031660581348773305, 0.09105937379881315, 0.9102078315323474]
---------
Input: [6.4 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.023344162096812622, 0.971248832147348, 0.026546180749920777]
---------
Input: [4.8 3.  1.4 0.3] Expected: [1. 0. 0.] Produced: [0.9874235009818471, 0.015580954481341431, 0.007239346650886765]
---------
Epoch 1000 RMSE = 0.14824140526748017
Epoch 1100 RMSE = 0.11600651403629096
Epoch 1200 RMSE = 0.15636924382083306
Epoch 1300 RMSE = 0.13824937903768478
Epoch 1400 RMSE = 0.10712023713966332
Epoch 1500 RMSE = 0.12609825945784417
Epoch 1600 RMSE = 0.13837753888257492
Epoch 1700 RMSE = 0.14982472128489713
Epoch 1800 RMSE = 0.13408524867591473
Epoch 1900 RMSE = 0.1209891992072135
Input: [4.3 3.  1.1 0.1] Expected: [1. 0. 0.] Produced: [0.9954699310372137, 0.005554463038982691, 0.01239099049126296]
---------
Input: [7.2 3.6 6.1 2.5] Expected: [0. 0. 1.] Produced: [0.0016682409027358649, 0.05842734685157882, 0.942033004347986]
---------
Input: [6.  3.  4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0018511365473633115, 0.07797813967289188, 0.9223472203371025]
---------
Input: [5.6 2.9 3.6 1.3] Expected: [0. 1. 0.] Produced: [0.016497949472251364, 0.9794253740113296, 0.017449071547068422]
---------
Input: [4.9 2.4 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.016624603953461128, 0.9791865785652094, 0.01786085621874341]
---------
Input: [6.9 3.2 5.7 2.3] Expected: [0. 0. 1.] Produced: [0.0016688197184814044, 0.058413911134571936, 0.9420423542349092]
---------
Input: [6.2 2.8 4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0018884020230189905, 0.08264328954917056, 0.9177086663018978]
---------
Input: [6.7 3.3 5.7 2.1] Expected: [0. 0. 1.] Produced: [0.0016702149932476035, 0.05847580968226804, 0.941976893009447]
---------
Input: [5.  3.2 1.2 0.2] Expected: [1. 0. 0.] Produced: [0.9954734204066954, 0.005548129852089202, 0.012116401844002552]
---------
Input: [5.8 2.7 3.9 1.2] Expected: [0. 1. 0.] Produced: [0.015344975472492159, 0.9807363409200633, 0.01791435507121115]
---------
Input: [6.6 2.9 4.6 1.3] Expected: [0. 1. 0.] Produced: [0.014825070398771224, 0.9812890131163234, 0.017998818819532105]
---------
Input: [5.8 2.7 4.1 1. ] Expected: [0. 1. 0.] Produced: [0.015184211521650243, 0.9809372708910786, 0.01800609077904051]
---------
Input: [6.2 3.4 5.4 2.3] Expected: [0. 0. 1.] Produced: [0.0016689790024331558, 0.058297794104466394, 0.9421441306197635]
---------
Input: [5.9 3.  5.1 1.8] Expected: [0. 0. 1.] Produced: [0.0016758580975234983, 0.058801547371091896, 0.9416225381107095]
---------
Input: [6.4 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.015025923810265482, 0.9811411036041435, 0.017908976188552674]
---------
Input: [6.5 3.  5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0016778187510976767, 0.05915200319910717, 0.9412969468974427]
---------
Input: [5.1 3.5 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9952842216398378, 0.005766424164474263, 0.01167242545563047]
---------
Input: [5.4 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9955091864641663, 0.005505770729711804, 0.012096348592540046]
---------
Input: [6.2 2.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.003516565601497378, 0.380470343723932, 0.6179383514582778]
---------
Input: [4.8 3.1 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9897055655012106, 0.012101441257463949, 0.006744505602109794]
---------
Input: [5.9 3.  4.2 1.5] Expected: [0. 1. 0.] Produced: [0.014928490549204473, 0.9810398690963235, 0.018020094560384836]
---------
Input: [6.8 2.8 4.8 1.4] Expected: [0. 1. 0.] Produced: [0.014723388930957333, 0.9814702133879959, 0.017917793530728213]
---------
Input: [6.5 2.8 4.6 1.5] Expected: [0. 1. 0.] Produced: [0.01465075542046538, 0.9810674251081415, 0.018278698746683728]
---------
Input: [7.7 3.8 6.7 2.2] Expected: [0. 0. 1.] Produced: [0.0016764245273101091, 0.0598119939147032, 0.9406558882879302]
---------
Input: [6.8 3.  5.5 2.1] Expected: [0. 0. 1.] Produced: [0.0016888162688092308, 0.060995101294864365, 0.9394620135282984]
---------
Input: [6.3 2.8 5.1 1.5] Expected: [0. 0. 1.] Produced: [0.0039882312766251465, 0.4798444992183503, 0.5181854835167642]
---------
Input: [4.4 2.9 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9906408333164148, 0.011036526872822429, 0.0073729691937006354]
---------
Input: [6.2 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.01497515269191543, 0.9811296420591196, 0.01797095487900411]
---------
Input: [5.6 2.5 3.9 1.1] Expected: [0. 1. 0.] Produced: [0.015143277834970548, 0.9807757012405323, 0.018139529882089573]
---------
Input: [5.1 3.8 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9950006031582826, 0.006094839731137978, 0.011095031195630205]
---------
Input: [6.7 3.1 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0016681281587830393, 0.0583551390392292, 0.9421027685807328]
---------
Input: [6.7 3.1 4.7 1.5] Expected: [0. 1. 0.] Produced: [0.014752148915078006, 0.9811804233775571, 0.01812152230326723]
---------
Input: [7.4 2.8 6.1 1.9] Expected: [0. 0. 1.] Produced: [0.0016701695874696129, 0.05857127773544205, 0.9418922372622671]
---------
Input: [6.9 3.1 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.014664633979466152, 0.9810913664299342, 0.01827044965785884]
---------
Input: [5.7 2.5 5.  2. ] Expected: [0. 0. 1.] Produced: [0.0016687073910098894, 0.05829602406655674, 0.9421542223545186]
---------
Input: [5.  3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9940158152473753, 0.00722423126381416, 0.009679625225623118]
---------
Input: [6.3 2.5 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.0022785711238738496, 0.1388823078892064, 0.8613558143806564]
---------
Input: [6.5 3.2 5.1 2. ] Expected: [0. 0. 1.] Produced: [0.0019923874236264147, 0.09716640831853825, 0.9031916796443904]
---------
Input: [5.7 3.8 1.7 0.3] Expected: [1. 0. 0.] Produced: [0.9940576186339695, 0.007182742999591044, 0.009429802511446048]
---------
Input: [4.6 3.1 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9912473650251822, 0.010370188203274957, 0.00754326112411704]
---------
Input: [5.  2.  3.5 1. ] Expected: [0. 1. 0.] Produced: [0.014949755161647286, 0.9794671833457552, 0.019331296518020246]
---------
Input: [7.2 3.2 6.  1.8] Expected: [0. 0. 1.] Produced: [0.0017451833501337, 0.06698894892519756, 0.9334828088810553]
---------
Input: [6.4 3.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.01489688184278865, 0.981286083754747, 0.01787964605176114]
---------
Input: [6.6 3.  4.4 1.4] Expected: [0. 1. 0.] Produced: [0.01497498693035239, 0.9813090638559341, 0.01779414580873719]
---------
Input: [5.3 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9954422517367091, 0.005588571028126684, 0.011947496234675867]
---------
Input: [5.2 3.4 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9950166151267393, 0.006081515547608597, 0.011136203578769766]
---------
Input: [5.  2.3 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.01642188972026155, 0.9795549241140263, 0.017799081803946275]
---------
Input: [5.2 2.7 3.9 1.4] Expected: [0. 1. 0.] Produced: [0.012575587787383349, 0.9674524599981674, 0.031003277267078383]
---------
Input: [6.7 2.5 5.8 1.8] Expected: [0. 0. 1.] Produced: [0.0016681066150135608, 0.058792287535287216, 0.9416737637723624]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9930533084539118, 0.008331140181286213, 0.008911336594395733]
---------
Input: [5.6 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.006468368923714914, 0.8007885473378236, 0.19575400632761997]
---------
Input: [6.3 2.7 4.9 1.8] Expected: [0. 0. 1.] Produced: [0.0019329127317363319, 0.08931135696580976, 0.9110879462981457]
---------
Input: [5.4 3.9 1.3 0.4] Expected: [1. 0. 0.] Produced: [0.9961035568658996, 0.004818521925252212, 0.013533704586404328]
---------
Input: [5.1 3.8 1.5 0.3] Expected: [1. 0. 0.] Produced: [0.9950821781424658, 0.006006561657173073, 0.011139704557375757]
---------
Input: [5.  3.  1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9890423244309327, 0.012833316077259602, 0.00644490310468371]
---------
Input: [6.4 2.7 5.3 1.9] Expected: [0. 0. 1.] Produced: [0.0016737677536978349, 0.05934615356929311, 0.9411118222881452]
---------
Input: [4.7 3.2 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9945421181612562, 0.006630374029204258, 0.010475932852703228]
---------
Input: [6.7 3.3 5.7 2.5] Expected: [0. 0. 1.] Produced: [0.0016679734912609598, 0.05882087595542455, 0.9416427692464534]
---------
Input: [4.8 3.  1.4 0.3] Expected: [1. 0. 0.] Produced: [0.9910978473183123, 0.01053664482409556, 0.00719739872155737]
---------
Input: [5.7 2.8 4.5 1.3] Expected: [0. 1. 0.] Produced: [0.012041106137052112, 0.9653126994088369, 0.03362139702717026]
---------
Input: [5.  3.4 1.6 0.4] Expected: [1. 0. 0.] Produced: [0.9902056877103448, 0.011530146707323382, 0.0064452030028786045]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9930544641675005, 0.008331867142985136, 0.008909972582931627]
---------
Input: [5.1 3.3 1.7 0.5] Expected: [1. 0. 0.] Produced: [0.9838805256192678, 0.018478234300403046, 0.00447939671682784]
---------
Input: [5.1 3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9941300240046504, 0.007102569679927475, 0.009789254470506709]
---------
Input: [5.6 2.7 4.2 1.3] Expected: [0. 1. 0.] Produced: [0.014146557073621342, 0.9781449637615977, 0.02098246346402211]
---------
Input: [5.  3.6 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9955217932475621, 0.0054971844236038625, 0.012174437103922985]
---------
Input: [5.8 2.8 5.1 2.4] Expected: [0. 0. 1.] Produced: [0.0016681895098374423, 0.05878627536946549, 0.9416706548199174]
---------
Input: [5.7 4.4 1.5 0.4] Expected: [1. 0. 0.] Produced: [0.996440704447884, 0.004422669676316198, 0.0146492465883611]
---------
Input: [6.  3.4 4.5 1.6] Expected: [0. 1. 0.] Produced: [0.01467524433882597, 0.9804464568911746, 0.01868714140157609]
---------
Input: [7.7 3.  6.1 2.3] Expected: [0. 0. 1.] Produced: [0.0016711349996196142, 0.059143911106915734, 0.941323567454167]
---------
Input: [4.6 3.2 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9934642636177827, 0.00786380683590159, 0.009180664496613445]
---------
Input: [7.3 2.9 6.3 1.8] Expected: [0. 0. 1.] Produced: [0.0016703085052091528, 0.059036091791117504, 0.9414318814497723]
---------
Input: [7.  3.2 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.014892214955638286, 0.98142679369583, 0.017785367109824105]
---------
Input: [6.1 2.9 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.013060945182687923, 0.9731605057161309, 0.026010097441095115]
---------
Input: [6.3 3.3 6.  2.5] Expected: [0. 0. 1.] Produced: [0.0016675781259270518, 0.058752740926662744, 0.941714017387274]
---------
Input: [5.5 2.4 3.8 1.1] Expected: [0. 1. 0.] Produced: [0.0151997819593124, 0.9808816651468277, 0.017998137187369815]
---------
Input: [6.4 3.1 5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0016918666139096072, 0.06110512389391539, 0.9393489750870105]
---------
Input: [4.9 2.5 4.5 1.7] Expected: [0. 0. 1.] Produced: [0.0016721231896636188, 0.058641423660802265, 0.9417607190098489]
---------
Input: [4.5 2.3 1.3 0.3] Expected: [1. 0. 0.] Produced: [0.980666562658951, 0.021983421221487717, 0.005016359761731141]
---------
Input: [5.9 3.2 4.8 1.8] Expected: [0. 1. 0.] Produced: [0.002406026581489654, 0.16027785420971383, 0.8395982319472192]
---------
Input: [6.3 3.4 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0016681592156951072, 0.05927914714891196, 0.9411811935369375]
---------
Input: [5.  3.5 1.6 0.6] Expected: [1. 0. 0.] Produced: [0.9875818803996952, 0.014455558880716998, 0.0051480704363917225]
---------
Input: [5.6 3.  4.1 1.3] Expected: [0. 1. 0.] Produced: [0.01513484155557478, 0.9810873051015565, 0.017809516188715915]
---------
Input: [6.1 3.  4.9 1.8] Expected: [0. 0. 1.] Produced: [0.00246749131076235, 0.17321364439709816, 0.8267747298300191]
---------
Input: [5.5 3.5 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9960317865062137, 0.004905022971672436, 0.01342465349063306]
---------
Input: [5.4 3.4 1.7 0.2] Expected: [1. 0. 0.] Produced: [0.9922026774609466, 0.009302195753773047, 0.007829859411225347]
---------
Input: [6.5 3.  5.2 2. ] Expected: [0. 0. 1.] Produced: [0.0017724590019820886, 0.07028605786685925, 0.9301558092750783]
---------
Input: [6.9 3.1 5.1 2.3] Expected: [0. 0. 1.] Produced: [0.002063840023843585, 0.10786480299633859, 0.8924912332750122]
---------
Input: [6.4 2.8 5.6 2.1] Expected: [0. 0. 1.] Produced: [0.0016679927265483927, 0.059056956999185846, 0.9414061939239299]
---------
Input: [5.1 2.5 3.  1.1] Expected: [0. 1. 0.] Produced: [0.020658947475059312, 0.9747012999482749, 0.016281897141607795]
---------
Input: [5.2 3.5 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9946756049299128, 0.006479100656999863, 0.010536971298803903]
---------
Input: [4.7 3.2 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9905983970461061, 0.01110759772024834, 0.007147379707387769]
---------
Input: [4.8 3.4 1.9 0.2] Expected: [1. 0. 0.] Produced: [0.9855377300249368, 0.016710073397625674, 0.005528226286979836]
---------
Input: [4.6 3.6 1.  0.2] Expected: [1. 0. 0.] Produced: [0.996835548340926, 0.00395832878179618, 0.016432689537536335]
---------
Input: [6.1 2.8 4.7 1.2] Expected: [0. 1. 0.] Produced: [0.014494669889843914, 0.9804541573327726, 0.01890007443319825]
---------
Input: [5.8 2.7 5.1 1.9] Expected: [0. 0. 1.] Produced: [0.0016695780293183044, 0.05908445206663902, 0.9413649739013566]
---------
Input: [7.7 2.6 6.9 2.3] Expected: [0. 0. 1.] Produced: [0.0016673637275989484, 0.05901995832452759, 0.9414477128378629]
---------
Input: [5.5 2.4 3.7 1. ] Expected: [0. 1. 0.] Produced: [0.015519954527428902, 0.9807171107002575, 0.01783672748904432]
---------
Input: [6.7 3.1 4.4 1.4] Expected: [0. 1. 0.] Produced: [0.01504734338154542, 0.9812912254381359, 0.017707047293868336]
---------
Input: [6.4 3.2 5.3 2.3] Expected: [0. 0. 1.] Produced: [0.0016718794479663724, 0.05932616062544368, 0.9411177481605277]
---------
Input: [4.9 3.  1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9926150256404757, 0.008833605146977954, 0.008359963611459256]
---------
Input: [5.4 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.004701209866959901, 0.6004251636383442, 0.39589340624551345]
---------
Input: [5.  3.3 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9944462675045567, 0.006750080196338364, 0.01024045000960045]
---------
Input: [7.9 3.8 6.4 2. ] Expected: [0. 0. 1.] Produced: [0.009311595206988621, 0.9287440826554498, 0.06980013742058559]
---------
Input: [6.  2.2 5.  1.5] Expected: [0. 0. 1.] Produced: [0.0017026381162496072, 0.06263387166430805, 0.9378223813287448]
---------
Epoch 2000 RMSE = 0.14504717864832523
Epoch 2100 RMSE = 0.14228376542020335
Epoch 2200 RMSE = 0.09812521647136327
Epoch 2300 RMSE = 0.12820492165934677
Epoch 2400 RMSE = 0.1331482259923842
Epoch 2500 RMSE = 0.12448849050574556
Epoch 2600 RMSE = 0.1295320038837552
Epoch 2700 RMSE = 0.1397179170059547
Epoch 2800 RMSE = 0.1309370923436312
Epoch 2900 RMSE = 0.14620219114040528
Input: [5.6 2.5 3.9 1.1] Expected: [0. 1. 0.] Produced: [0.011965416995751698, 0.9835955193078543, 0.015626984802478087]
---------
Input: [5.  2.  3.5 1. ] Expected: [0. 1. 0.] Produced: [0.01166068742063863, 0.9817333517335802, 0.01736545809421112]
---------
Input: [5.9 3.  5.1 1.8] Expected: [0. 0. 1.] Produced: [0.0013092855328637508, 0.05616863381421496, 0.9441012490324684]
---------
Input: [6.1 3.  4.9 1.8] Expected: [0. 0. 1.] Produced: [0.001352051452503602, 0.061590690225959185, 0.9386319302064113]
---------
Input: [5.  3.  1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9906998799602245, 0.010409037621814107, 0.005905353764668911]
---------
Input: [5.1 3.5 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9960944479979376, 0.004629092136383802, 0.010906478902463546]
---------
Input: [6.3 2.8 5.1 1.5] Expected: [0. 0. 1.] Produced: [0.0014662667862706052, 0.0783454153667624, 0.9219092316561969]
---------
Input: [7.7 3.  6.1 2.3] Expected: [0. 0. 1.] Produced: [0.0013070978746738122, 0.05603155224176885, 0.9442701392922469]
---------
Input: [5.7 4.4 1.5 0.4] Expected: [1. 0. 0.] Produced: [0.9970457395518931, 0.003568670439564021, 0.01365453342832923]
---------
Input: [7.2 3.6 6.1 2.5] Expected: [0. 0. 1.] Produced: [0.0013070453166510898, 0.05600269231843577, 0.9442968743202571]
---------
Input: [5.  3.5 1.6 0.6] Expected: [1. 0. 0.] Produced: [0.9893923742634512, 0.011733740664780645, 0.004744235849249396]
---------
Input: [6.2 2.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.0020085023518373406, 0.18634729827539245, 0.813441662168457]
---------
Input: [5.1 3.8 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9958410969592881, 0.004912716625820197, 0.010323001811021347]
---------
Input: [5.4 3.9 1.3 0.4] Expected: [1. 0. 0.] Produced: [0.996786860762395, 0.003861631113643112, 0.012694403899896844]
---------
Input: [5.5 3.5 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9967185617210819, 0.003938693294910118, 0.012558151471301196]
---------
Input: [5.8 2.7 4.1 1. ] Expected: [0. 1. 0.] Produced: [0.01198649154697053, 0.9837945504531792, 0.015452932622130598]
---------
Input: [6.9 3.2 5.7 2.3] Expected: [0. 0. 1.] Produced: [0.0013073366246570846, 0.056595715709785964, 0.9437025473628475]
---------
Input: [4.8 3.1 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9912304660144341, 0.009870465651816918, 0.006149305399972405]
---------
Input: [4.7 3.2 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9954659937395437, 0.005326374642540528, 0.009765973533329407]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9941921610325142, 0.0067148956664876015, 0.008249821175566227]
---------
Input: [6.  3.4 4.5 1.6] Expected: [0. 1. 0.] Produced: [0.011656217757143122, 0.9834168412397153, 0.015995598151901266]
---------
Input: [6.4 3.2 5.3 2.3] Expected: [0. 0. 1.] Produced: [0.0013077692252711796, 0.05657621535487185, 0.9437081652948945]
---------
Input: [5.7 2.5 5.  2. ] Expected: [0. 0. 1.] Produced: [0.0013074718378967359, 0.05652859937824596, 0.9437636201511104]
---------
Input: [5.  2.3 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.012908635225498816, 0.9825972050608323, 0.015297649850400172]
---------
Input: [6.7 3.1 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0013071227386276616, 0.056528509848599494, 0.9437697688317869]
---------
Input: [5.9 3.2 4.8 1.8] Expected: [0. 1. 0.] Produced: [0.001573264882995646, 0.09621464416425049, 0.9038483301807324]
---------
Input: [7.  3.2 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.01178219936931233, 0.9841166133368253, 0.015350596454117581]
---------
Input: [6.1 2.8 4.7 1.2] Expected: [0. 1. 0.] Produced: [0.01136948613550444, 0.982727329502023, 0.016854037143533538]
---------
Input: [5.8 2.8 5.1 2.4] Expected: [0. 0. 1.] Produced: [0.0013072389737222708, 0.05690851367854405, 0.9433874258138043]
---------
Input: [6.8 2.8 4.8 1.4] Expected: [0. 1. 0.] Produced: [0.01165080865657162, 0.984085435776656, 0.015528445071511309]
---------
Input: [6.7 3.1 4.4 1.4] Expected: [0. 1. 0.] Produced: [0.011895941341542636, 0.9839685846592222, 0.015316759565373848]
---------
Input: [4.3 3.  1.1 0.1] Expected: [1. 0. 0.] Produced: [0.9962072523304589, 0.004512878264502183, 0.011423386029194916]
---------
Input: [4.6 3.2 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9945237200830763, 0.006358948879827094, 0.00848811430780314]
---------
Input: [6.3 2.7 4.9 1.8] Expected: [0. 0. 1.] Produced: [0.001393969164962475, 0.06856960415832851, 0.9316846336420014]
---------
Input: [6.2 2.8 4.8 1.8] Expected: [0. 0. 1.] Produced: [0.001535958100147881, 0.09074333295790085, 0.9094393606217931]
---------
Input: [5.1 3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9951273363783892, 0.005698879151726324, 0.009136380044583316]
---------
Input: [4.7 3.2 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9919811071457552, 0.009084694407147674, 0.006508046655930035]
---------
Input: [5.6 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.0053954223432149334, 0.8401952467986137, 0.15744865040003467]
---------
Input: [5.7 3.8 1.7 0.3] Expected: [1. 0. 0.] Produced: [0.9951178691701714, 0.00570751707180048, 0.008900783009426445]
---------
Input: [5.2 2.7 3.9 1.4] Expected: [0. 1. 0.] Produced: [0.01077027901975102, 0.9781140374757891, 0.020999233315855785]
---------
Input: [6.5 3.  5.2 2. ] Expected: [0. 0. 1.] Produced: [0.0013266601206557136, 0.05936625071019421, 0.9409134061025984]
---------
Input: [5.5 2.4 3.8 1.1] Expected: [0. 1. 0.] Produced: [0.012007976974343654, 0.9837647271718115, 0.015431103131941216]
---------
Input: [6.4 2.8 5.6 2.1] Expected: [0. 0. 1.] Produced: [0.0013070437425472712, 0.05690157098895575, 0.943402479193357]
---------
Input: [6.7 2.5 5.8 1.8] Expected: [0. 0. 1.] Produced: [0.001307024308346463, 0.056891205902384466, 0.9434148048697419]
---------
Input: [5.4 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.0026303548604597283, 0.35025760396978034, 0.6479217954208468]
---------
Input: [7.3 2.9 6.3 1.8] Expected: [0. 0. 1.] Produced: [0.0013090855008892981, 0.05782833669957308, 0.94248227815203]
---------
Input: [5.  3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9950151460647514, 0.005831245944579812, 0.008976947014030259]
---------
Input: [6.4 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.011878516755120002, 0.9841053715916649, 0.015236920823431734]
---------
Input: [5.4 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9962798708065128, 0.004434414994040907, 0.011271825200219054]
---------
Input: [6.2 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.01185262785016417, 0.9841356663713168, 0.015250521116367382]
---------
Input: [6.4 3.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.0118008321745357, 0.9841984449101863, 0.015238860765050908]
---------
Input: [5.6 2.9 3.6 1.3] Expected: [0. 1. 0.] Produced: [0.012986196443696045, 0.9826941654332739, 0.014814847263066802]
---------
Input: [5.  3.6 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9962719706471446, 0.004443629893347227, 0.01130199619343833]
---------
Input: [7.9 3.8 6.4 2. ] Expected: [0. 0. 1.] Produced: [0.005755599754265687, 0.870217511438639, 0.12836445466917235]
---------
Input: [4.9 2.5 4.5 1.7] Expected: [0. 0. 1.] Produced: [0.001309251996829813, 0.05710052559749329, 0.9431478809779513]
---------
Input: [6.7 3.3 5.7 2.1] Expected: [0. 0. 1.] Produced: [0.0013084706355372382, 0.05732186364804183, 0.9429749856072783]
---------
Input: [5.1 2.5 3.  1.1] Expected: [0. 1. 0.] Produced: [0.0162197243135808, 0.978497314708269, 0.01388648387000988]
---------
Input: [5.  3.3 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9953875884838712, 0.005417240817584229, 0.009559444105570442]
---------
Input: [7.7 2.6 6.9 2.3] Expected: [0. 0. 1.] Produced: [0.0013067597535115255, 0.05714916323131262, 0.9431579212909877]
---------
Input: [5.2 3.5 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9955804845591889, 0.005204496355780893, 0.00983333728484447]
---------
Input: [5.1 3.8 1.5 0.3] Expected: [1. 0. 0.] Produced: [0.9959237804341169, 0.004825041789538099, 0.010407955512805496]
---------
Input: [6.4 3.1 5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0013142484287997744, 0.05799874168537129, 0.9422933002557784]
---------
Input: [5.  3.4 1.6 0.4] Expected: [1. 0. 0.] Produced: [0.9917116007546011, 0.009363231060148936, 0.0059300640674416925]
---------
Input: [6.5 2.8 4.6 1.5] Expected: [0. 1. 0.] Produced: [0.011599856918137343, 0.9838048631719303, 0.01577775203175615]
---------
Input: [5.  3.2 1.2 0.2] Expected: [1. 0. 0.] Produced: [0.9962505251554468, 0.004464042940634295, 0.011294453695120076]
---------
Input: [6.6 3.  4.4 1.4] Expected: [0. 1. 0.] Produced: [0.011841344905360228, 0.9840647495195867, 0.01531329086314835]
---------
Input: [7.4 2.8 6.1 1.9] Expected: [0. 0. 1.] Produced: [0.0013081138121301893, 0.05727442758245309, 0.9430302528991898]
---------
Input: [7.7 3.8 6.7 2.2] Expected: [0. 0. 1.] Produced: [0.001307740682973091, 0.057218560075022226, 0.9430862293652891]
---------
Input: [6.8 3.  5.5 2.1] Expected: [0. 0. 1.] Produced: [0.001310151018152081, 0.05745462601897621, 0.9428397673138865]
---------
Input: [6.  2.2 5.  1.5] Expected: [0. 0. 1.] Produced: [0.0013106078870310078, 0.0574050924969453, 0.9428851578401887]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9941781894044084, 0.006737148086885652, 0.0082226753110368]
---------
Input: [5.9 3.  4.2 1.5] Expected: [0. 1. 0.] Produced: [0.011816728750314394, 0.9838610924332039, 0.015484765678741864]
---------
Input: [6.7 3.3 5.7 2.5] Expected: [0. 0. 1.] Produced: [0.0013069535030415984, 0.05704498045634524, 0.943254120187934]
---------
Input: [4.6 3.1 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9925644839175551, 0.008469306011545209, 0.006881490261931396]
---------
Input: [5.6 2.7 4.2 1.3] Expected: [0. 1. 0.] Produced: [0.011400064606932441, 0.9823578858965135, 0.017085595262256652]
---------
Input: [6.6 2.9 4.6 1.3] Expected: [0. 1. 0.] Produced: [0.011743504109397278, 0.9841667946567714, 0.015370354555643707]
---------
Input: [5.2 3.4 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9958785032905884, 0.004875287257321009, 0.010415447310749082]
---------
Input: [6.3 2.5 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.0020426386345575686, 0.19826978306179757, 0.8016185887721997]
---------
Input: [5.8 2.7 5.1 1.9] Expected: [0. 0. 1.] Produced: [0.0013076778288797462, 0.0576575518341063, 0.9426340126007002]
---------
Input: [4.8 3.  1.4 0.3] Expected: [1. 0. 0.] Produced: [0.9925111654532355, 0.008536071986783198, 0.006629134083070566]
---------
Input: [5.7 2.8 4.5 1.3] Expected: [0. 1. 0.] Produced: [0.010929829782786107, 0.9804947564347561, 0.019016593460599664]
---------
Input: [6.2 3.4 5.4 2.3] Expected: [0. 0. 1.] Produced: [0.0013077169960390392, 0.057671832562355564, 0.9426165553331378]
---------
Input: [4.4 2.9 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9920132459688201, 0.009075849690672839, 0.0066711146340242]
---------
Input: [5.6 3.  4.1 1.3] Expected: [0. 1. 0.] Produced: [0.011964297621217063, 0.9839572527199084, 0.015253433629151136]
---------
Input: [4.9 2.4 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.013062312559410035, 0.9825954379544414, 0.015081123301019083]
---------
Input: [5.1 3.3 1.7 0.5] Expected: [1. 0. 0.] Produced: [0.985894190657445, 0.015429582109239944, 0.0039934157284223755]
---------
Input: [6.7 3.1 4.7 1.5] Expected: [0. 1. 0.] Produced: [0.011722714291946112, 0.9842877320325973, 0.015266879450917753]
---------
Input: [6.5 3.  5.5 1.8] Expected: [0. 0. 1.] Produced: [0.00133011536320853, 0.06064338393896864, 0.93965274324532]
---------
Input: [4.5 2.3 1.3 0.3] Expected: [1. 0. 0.] Produced: [0.9830150903749799, 0.018445939883819484, 0.00441609034859651]
---------
Input: [6.9 3.1 5.1 2.3] Expected: [0. 0. 1.] Produced: [0.0016376588383607957, 0.11063648870984663, 0.8895593280170577]
---------
Input: [6.3 3.3 6.  2.5] Expected: [0. 0. 1.] Produced: [0.0013067685648309663, 0.057545363455034526, 0.9427592215929155]
---------
Input: [4.9 3.  1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9938268393674387, 0.007123333837293852, 0.007741440486535047]
---------
Input: [5.8 2.7 3.9 1.2] Expected: [0. 1. 0.] Produced: [0.012112066702495136, 0.9837901366337022, 0.015219912972346265]
---------
Input: [5.4 3.4 1.7 0.2] Expected: [1. 0. 0.] Produced: [0.9934840954016343, 0.007489375096301354, 0.007265433355961531]
---------
Input: [5.3 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9962248177559926, 0.004495025053183052, 0.011145625316224093]
---------
Input: [5.5 2.4 3.7 1. ] Expected: [0. 1. 0.] Produced: [0.012228274288277075, 0.9836434150469374, 0.01527670646413459]
---------
Input: [6.  3.  4.8 1.8] Expected: [0. 0. 1.] Produced: [0.001969707608428141, 0.18137265389902793, 0.8184268180043477]
---------
Input: [6.3 3.4 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0013070068748591657, 0.05739229492602614, 0.9429040274616203]
---------
Input: [6.1 2.9 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.011169974100825522, 0.9818938390206013, 0.017672962606433546]
---------
Input: [7.2 3.2 6.  1.8] Expected: [0. 0. 1.] Produced: [0.0013994028388100237, 0.07025123851476109, 0.9300558832652205]
---------
Input: [6.5 3.2 5.1 2. ] Expected: [0. 0. 1.] Produced: [0.0018545751538580092, 0.15493654214005775, 0.8450877917952915]
---------
Input: [4.6 3.6 1.  0.2] Expected: [1. 0. 0.] Produced: [0.9973212528215003, 0.0032620452147314885, 0.014988905515880058]
---------
Input: [6.9 3.1 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.011683125917555185, 0.9842599075323224, 0.01534438872995434]
---------
Input: [6.4 2.7 5.3 1.9] Expected: [0. 0. 1.] Produced: [0.0013094467116283926, 0.05753020279978913, 0.942762025583019]
---------
Input: [4.8 3.4 1.9 0.2] Expected: [1. 0. 0.] Produced: [0.9872271870246853, 0.01407452788013921, 0.004862509838905336]
---------
Epoch 3000 RMSE = 0.14959525009188418
Epoch 3100 RMSE = 0.149267099436157
Epoch 3200 RMSE = 0.14897424904330167
Epoch 3300 RMSE = 0.11302410127813803
Epoch 3400 RMSE = 0.13498526793150423
Epoch 3500 RMSE = 0.1448002543262192
Epoch 3600 RMSE = 0.12432059947952802
Epoch 3700 RMSE = 0.14599487794218227
Epoch 3800 RMSE = 0.13878137710091296
Epoch 3900 RMSE = 0.11874607157931584
Input: [7.7 3.8 6.7 2.2] Expected: [0. 0. 1.] Produced: [0.0011082339745644714, 0.058192006957521504, 0.9419900220366295]
---------
Input: [5.1 2.5 3.  1.1] Expected: [0. 1. 0.] Produced: [0.013704204972194144, 0.9803449881402662, 0.012807114784508137]
---------
Input: [4.9 2.5 4.5 1.7] Expected: [0. 0. 1.] Produced: [0.001109010752954503, 0.057972367136795024, 0.9421571795399364]
---------
Input: [5.7 4.4 1.5 0.4] Expected: [1. 0. 0.] Produced: [0.9973991108952498, 0.0031007294981630702, 0.012538240620764335]
---------
Input: [5.8 2.7 5.1 1.9] Expected: [0. 0. 1.] Produced: [0.001107851172821013, 0.05800988474302275, 0.942158792688856]
---------
Input: [6.4 3.1 5.5 1.8] Expected: [0. 0. 1.] Produced: [0.001114609785124254, 0.05906715710500971, 0.9411020917667604]
---------
Input: [6.7 3.3 5.7 2.1] Expected: [0. 0. 1.] Produced: [0.0011087523097064182, 0.05816352272127493, 0.942010680055968]
---------
Input: [6.1 2.8 4.7 1.2] Expected: [0. 1. 0.] Produced: [0.009915702749206818, 0.9851916419735522, 0.0145222603007493]
---------
Input: [5.  3.2 1.2 0.2] Expected: [1. 0. 0.] Produced: [0.9967255457881942, 0.0038332354105630774, 0.010481103222678938]
---------
Input: [6.4 2.8 5.6 2.1] Expected: [0. 0. 1.] Produced: [0.0011074537062602037, 0.05796657075957263, 0.9422127317527023]
---------
Input: [5.7 2.8 4.5 1.3] Expected: [0. 1. 0.] Produced: [0.009346564320135255, 0.9820587775594193, 0.017582350478738837]
---------
Input: [6.5 3.2 5.1 2. ] Expected: [0. 0. 1.] Produced: [0.0015083542678062517, 0.1408979068671017, 0.8590482306607721]
---------
Input: [5.5 2.4 3.7 1. ] Expected: [0. 1. 0.] Produced: [0.010393792951603432, 0.9848755271279994, 0.014222382885851122]
---------
Input: [5.  2.3 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.010945401092717194, 0.9841145229461615, 0.01407573909231859]
---------
Input: [5.1 3.5 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9965798406845383, 0.003988885762818608, 0.01008500244389404]
---------
Input: [5.5 3.5 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9971252277509489, 0.0033997505616337456, 0.011599721719024634]
---------
Input: [5.  3.6 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9967336493084906, 0.003823581809174219, 0.010468024164428852]
---------
Input: [5.2 3.5 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9961470665862597, 0.004450931466810178, 0.009158989971095986]
---------
Input: [4.9 3.  1.4 0.2] Expected: [1. 0. 0.] Produced: [0.994603511885758, 0.006074833655569658, 0.007220204152723081]
---------
Input: [6.5 3.  5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0011147959030542668, 0.05897054652829877, 0.9411997986669149]
---------
Input: [4.8 3.  1.4 0.3] Expected: [1. 0. 0.] Produced: [0.9934448624337859, 0.007265646423295895, 0.006193597647144217]
---------
Input: [4.3 3.  1.1 0.1] Expected: [1. 0. 0.] Produced: [0.9966581365456383, 0.003907523256733263, 0.010505313060417892]
---------
Input: [5.9 3.  5.1 1.8] Expected: [0. 0. 1.] Produced: [0.0011118857169088302, 0.05838737544335385, 0.9417622221385517]
---------
Input: [6.3 3.4 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0011075001727597813, 0.0578264527650031, 0.9423476520522501]
---------
Input: [5.2 3.4 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9964076367277888, 0.0041729754148076385, 0.009694633994687044]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.994901710172959, 0.0057666355424933315, 0.007637992986495237]
---------
Input: [5.4 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.0029043579321515437, 0.5652478207913055, 0.4323024733931885]
---------
Input: [5.6 2.7 4.2 1.3] Expected: [0. 1. 0.] Produced: [0.010006484977707833, 0.9852578941383545, 0.01435356617887476]
---------
Input: [5.1 3.8 1.5 0.3] Expected: [1. 0. 0.] Produced: [0.9964382440544051, 0.00414238911623939, 0.009663001889562242]
---------
Input: [4.5 2.3 1.3 0.3] Expected: [1. 0. 0.] Produced: [0.9846928619200601, 0.016018074508740928, 0.004036056640158891]
---------
Input: [6.2 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.01009495529784265, 0.9853566672324825, 0.014161222337006731]
---------
Input: [5.5 2.4 3.8 1.1] Expected: [0. 1. 0.] Produced: [0.010224610730778764, 0.9851763760449731, 0.014178776202571484]
---------
Input: [5.8 2.7 4.1 1. ] Expected: [0. 1. 0.] Produced: [0.010198513891056618, 0.985216422896138, 0.014185333519029081]
---------
Input: [6.1 2.9 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.009864947453141551, 0.9850506616516583, 0.014663478802360633]
---------
Input: [6.2 3.4 5.4 2.3] Expected: [0. 0. 1.] Produced: [0.0011079595112292067, 0.058304339960756825, 0.9418645595262054]
---------
Input: [6.4 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.010115480191580933, 0.9853268246515395, 0.01415162956823184]
---------
Input: [4.6 3.6 1.  0.2] Expected: [1. 0. 0.] Produced: [0.9976145790220015, 0.002865374599686139, 0.01361359005195968]
---------
Input: [4.9 2.4 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.011071154989930033, 0.9840161411443581, 0.013964206454617323]
---------
Input: [4.6 3.1 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9934382225010369, 0.007288431707096883, 0.006355520133490991]
---------
Input: [5.4 3.9 1.3 0.4] Expected: [1. 0. 0.] Produced: [0.9971840442828882, 0.003336708337886457, 0.011717635129421748]
---------
Input: [5.6 3.  4.1 1.3] Expected: [0. 1. 0.] Produced: [0.010187017956573091, 0.9852218216956722, 0.01414186513967564]
---------
Input: [5.6 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.008698378260589697, 0.9776135409678074, 0.02194447584557775]
---------
Input: [6.9 3.2 5.7 2.3] Expected: [0. 0. 1.] Produced: [0.0011085405006371375, 0.05843108801363662, 0.9417486483614361]
---------
Input: [5.  3.4 1.6 0.4] Expected: [1. 0. 0.] Produced: [0.9927199232538784, 0.00800534860185014, 0.00551933733874486]
---------
Input: [5.9 3.2 4.8 1.8] Expected: [0. 1. 0.] Produced: [0.00300858580774044, 0.5971058440028427, 0.40098174825396943]
---------
Input: [5.3 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9966956877535262, 0.0038690011933594616, 0.010321429934732536]
---------
Input: [6.8 2.8 4.8 1.4] Expected: [0. 1. 0.] Produced: [0.009963524800285062, 0.9856011958590106, 0.014124847156495727]
---------
Input: [5.7 2.5 5.  2. ] Expected: [0. 0. 1.] Produced: [0.00110767283010288, 0.058646753942606954, 0.9415314957168044]
---------
Input: [5.  3.  1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9917603011182324, 0.0090087633485472, 0.005418837783286723]
---------
Input: [6.3 2.8 5.1 1.5] Expected: [0. 0. 1.] Produced: [0.007093894428314831, 0.958819152363588, 0.040641391718829724]
---------
Input: [7.9 3.8 6.4 2. ] Expected: [0. 0. 1.] Produced: [0.008453316368174707, 0.9761604770285599, 0.02350029093704871]
---------
Input: [4.6 3.2 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9951872260616841, 0.005473347818415166, 0.007846137561808726]
---------
Input: [5.1 3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9957413415407502, 0.004886881648954448, 0.008482858013881295]
---------
Input: [6.7 3.3 5.7 2.5] Expected: [0. 0. 1.] Produced: [0.0011074643404118568, 0.058460554084142816, 0.9417205846781304]
---------
Input: [6.6 2.9 4.6 1.3] Expected: [0. 1. 0.] Produced: [0.010010718518855726, 0.9854906408125941, 0.014164360467896667]
---------
Input: [5.8 2.8 5.1 2.4] Expected: [0. 0. 1.] Produced: [0.001107491077654161, 0.058423457611249514, 0.9417531044849415]
---------
Input: [6.3 3.3 6.  2.5] Expected: [0. 0. 1.] Produced: [0.0011072770249300394, 0.05841542308061653, 0.9417696616646709]
---------
Input: [6.6 3.  4.4 1.4] Expected: [0. 1. 0.] Produced: [0.010086999311045982, 0.9853809289347275, 0.014132991005310482]
---------
Input: [4.8 3.4 1.9 0.2] Expected: [1. 0. 0.] Produced: [0.9884326705565002, 0.012326001394914585, 0.004410084226366621]
---------
Input: [7.4 2.8 6.1 1.9] Expected: [0. 0. 1.] Produced: [0.001122155722438582, 0.06077419119036194, 0.9394106933350649]
---------
Input: [7.3 2.9 6.3 1.8] Expected: [0. 0. 1.] Produced: [0.0011111661220774161, 0.058993583404307416, 0.9411915567238555]
---------
Input: [5.6 2.9 3.6 1.3] Expected: [0. 1. 0.] Produced: [0.0110164057894519, 0.9840946718478798, 0.013731825881811145]
---------
Input: [6.  3.  4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0034391461771816315, 0.6952247252780277, 0.302988711920535]
---------
Input: [6.4 3.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.01005425950431393, 0.9853116982958546, 0.01424863778483237]
---------
Input: [6.4 3.2 5.3 2.3] Expected: [0. 0. 1.] Produced: [0.001108092609981384, 0.057816376629746466, 0.942348333191786]
---------
Input: [5.8 2.7 3.9 1.2] Expected: [0. 1. 0.] Produced: [0.01030193177768956, 0.9849701901752133, 0.014207251851159364]
---------
Input: [4.8 3.1 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9922266825377686, 0.00850754750434265, 0.005667464767372271]
---------
Input: [5.2 2.7 3.9 1.4] Expected: [0. 1. 0.] Produced: [0.009817466952614057, 0.983691037308556, 0.015726995362815018]
---------
Input: [5.6 2.5 3.9 1.1] Expected: [0. 1. 0.] Produced: [0.010200810256943225, 0.9851061036001434, 0.01427648018226268]
---------
Input: [6.2 2.8 4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0016737061018463845, 0.18519501556296022, 0.8145520438607213]
---------
Input: [7.7 2.6 6.9 2.3] Expected: [0. 0. 1.] Produced: [0.0011071979788202877, 0.05761343860466084, 0.9425703146366557]
---------
Input: [6.8 3.  5.5 2.1] Expected: [0. 0. 1.] Produced: [0.0011104536062503006, 0.058046481515706294, 0.9421263937077425]
---------
Input: [5.  3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.995647861187213, 0.004976507468559695, 0.008373435820828268]
---------
Input: [5.1 3.8 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.996353171343063, 0.004228551557981472, 0.009554559248944657]
---------
Input: [5.4 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9967516073730499, 0.003801821756608874, 0.010478013740540273]
---------
Input: [6.5 2.8 4.6 1.5] Expected: [0. 1. 0.] Produced: [0.009957006806756121, 0.9853075544414746, 0.014390468023111133]
---------
Input: [5.7 3.8 1.7 0.3] Expected: [1. 0. 0.] Produced: [0.9957644655406168, 0.0048497935555598375, 0.008345977941678864]
---------
Input: [6.3 2.5 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.002363877297156283, 0.4058070033557036, 0.5931921046136668]
---------
Input: [6.7 2.5 5.8 1.8] Expected: [0. 0. 1.] Produced: [0.0011074625874922016, 0.058248373365204706, 0.9419356196036666]
---------
Input: [5.1 3.3 1.7 0.5] Expected: [1. 0. 0.] Produced: [0.9873807349649196, 0.013295991302112274, 0.0036856853990128533]
---------
Input: [7.2 3.2 6.  1.8] Expected: [0. 0. 1.] Produced: [0.001428467319060939, 0.1221039534276124, 0.8780077906968408]
---------
Input: [6.5 3.  5.2 2. ] Expected: [0. 0. 1.] Produced: [0.001231799141088381, 0.07960470811510573, 0.9205370225364174]
---------
Input: [6.  3.4 4.5 1.6] Expected: [0. 1. 0.] Produced: [0.010046812133768481, 0.9853600565704048, 0.014202870550935338]
---------
Input: [6.9 3.1 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.009967089283500158, 0.9854943744581062, 0.014217886408764814]
---------
Input: [4.7 3.2 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9928978003925769, 0.007835387326310905, 0.005996139106994766]
---------
Input: [6.9 3.1 5.1 2.3] Expected: [0. 0. 1.] Produced: [0.002048078173258507, 0.3040477821198725, 0.6953439025771331]
---------
Input: [6.4 2.7 5.3 1.9] Expected: [0. 0. 1.] Produced: [0.0011098810621453894, 0.05815704447200626, 0.942016577344672]
---------
Input: [6.3 2.7 4.9 1.8] Expected: [0. 0. 1.] Produced: [0.0014748440172673183, 0.13217272717414916, 0.867826360330066]
---------
Input: [6.1 3.  4.9 1.8] Expected: [0. 0. 1.] Produced: [0.0016278116603099582, 0.1721426574094796, 0.8276598475744369]
---------
Input: [6.7 3.1 4.4 1.4] Expected: [0. 1. 0.] Produced: [0.010129493967199383, 0.9851941901484011, 0.014233685916253519]
---------
Input: [4.4 2.9 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9929452900830573, 0.007781481806233333, 0.006180549932450923]
---------
Input: [5.  3.3 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9959799755046648, 0.0046266558543367215, 0.008911405971536994]
---------
Input: [5.4 3.4 1.7 0.2] Expected: [1. 0. 0.] Produced: [0.9943156457228223, 0.006363781154290239, 0.006802168444512591]
---------
Input: [7.2 3.6 6.1 2.5] Expected: [0. 0. 1.] Produced: [0.0011073927079904783, 0.057608985256188835, 0.9425717387723158]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9949062494180465, 0.0057584753065857445, 0.007647510950113007]
---------
Input: [6.7 3.1 4.7 1.5] Expected: [0. 1. 0.] Produced: [0.009994238137305446, 0.9853763451529471, 0.014288708350260632]
---------
Input: [7.7 3.  6.1 2.3] Expected: [0. 0. 1.] Produced: [0.0011086552481767108, 0.05779641748425605, 0.9423858188412059]
---------
Input: [4.7 3.2 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9960298111677501, 0.004574492435052604, 0.009052230716087368]
---------
Input: [6.2 2.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.006695523938049186, 0.9499739654375512, 0.04935281442979051]
---------
Input: [7.  3.2 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.010038582785663979, 0.9853179063126366, 0.014271821219551726]
---------
Input: [5.9 3.  4.2 1.5] Expected: [0. 1. 0.] Produced: [0.010093770016665246, 0.9852118346150011, 0.014275158568549193]
---------
Input: [6.7 3.1 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0011073551339468937, 0.05756782062922589, 0.9426103878874843]
---------
Input: [6.  2.2 5.  1.5] Expected: [0. 0. 1.] Produced: [0.0011123971356352656, 0.058240137705941025, 0.9419296043171913]
---------
Input: [5.  3.5 1.6 0.6] Expected: [1. 0. 0.] Produced: [0.9906068574955602, 0.010086379958439369, 0.004394032930022374]
---------
Input: [5.  2.  3.5 1. ] Expected: [0. 1. 0.] Produced: [0.01024057494595439, 0.9849128746996632, 0.014426442257718617]
---------
Epoch 4000 RMSE = 0.1464235864104045
Epoch 4100 RMSE = 0.14209651572851126
Epoch 4200 RMSE = 0.12869260890549714
Epoch 4300 RMSE = 0.13159757513307593
Epoch 4400 RMSE = 0.14705328813622978
Epoch 4500 RMSE = 0.13723696167035979
Epoch 4600 RMSE = 0.14081281742959845
Epoch 4700 RMSE = 0.12994754900231184
Epoch 4800 RMSE = 0.1372786659200513
Epoch 4900 RMSE = 0.12367020704303597
Input: [5.6 2.9 3.6 1.3] Expected: [0. 1. 0.] Produced: [0.009708641320103035, 0.9848755415755511, 0.013187124835505567]
---------
Input: [4.5 2.3 1.3 0.3] Expected: [1. 0. 0.] Produced: [0.9858416282731863, 0.014393855388261668, 0.0037344406277470565]
---------
Input: [6.5 2.8 4.6 1.5] Expected: [0. 1. 0.] Produced: [0.008821283713334497, 0.9860892774711032, 0.013698233630093194]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9953784946243317, 0.005152890555952352, 0.007089453622316779]
---------
Input: [6.3 2.8 5.1 1.5] Expected: [0. 0. 1.] Produced: [0.001493259401669034, 0.1925448446926505, 0.8073614985861353]
---------
Input: [6.7 3.1 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0009767071825510405, 0.05768209319422082, 0.9424601289458812]
---------
Input: [5.1 3.8 1.5 0.3] Expected: [1. 0. 0.] Produced: [0.9967790514200936, 0.003703621874112132, 0.008970400692290936]
---------
Input: [4.9 2.5 4.5 1.7] Expected: [0. 0. 1.] Produced: [0.0009778364450900537, 0.05760770284889621, 0.9424904032037682]
---------
Input: [5.7 2.8 4.5 1.3] Expected: [0. 1. 0.] Produced: [0.007956718493523304, 0.9806004518408757, 0.01911578804551551]
---------
Input: [6.7 3.3 5.7 2.1] Expected: [0. 0. 1.] Produced: [0.0009769986307194788, 0.05768522912405798, 0.9424540582656158]
---------
Input: [6.2 2.8 4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0011500338652642456, 0.0931242459804946, 0.9069098617473262]
---------
Input: [5.7 4.4 1.5 0.4] Expected: [1. 0. 0.] Produced: [0.9976338559195523, 0.0027998221218858755, 0.011529272377332617]
---------
Input: [5.9 3.  5.1 1.8] Expected: [0. 0. 1.] Produced: [0.0009782213824318649, 0.05772968816680634, 0.9423877773487414]
---------
Input: [4.6 3.6 1.  0.2] Expected: [1. 0. 0.] Produced: [0.997814405087943, 0.002606504209874394, 0.012425308584012295]
---------
Input: [4.6 3.1 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9940203582127647, 0.0065127429291722775, 0.0059007635569634145]
---------
Input: [5.6 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.0051688998632436, 0.9263787373977117, 0.07275320335399998]
---------
Input: [6.5 3.2 5.1 2. ] Expected: [0. 0. 1.] Produced: [0.0011185607651150909, 0.08596470584144368, 0.9141054090117028]
---------
Input: [6.9 3.2 5.7 2.3] Expected: [0. 0. 1.] Produced: [0.0009767658297957316, 0.05756314559058527, 0.9425780418752682]
---------
Input: [5.3 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9970095802436646, 0.003462554204222336, 0.009571885927976066]
---------
Input: [5.8 2.8 5.1 2.4] Expected: [0. 0. 1.] Produced: [0.0009767757943917898, 0.05753378150747552, 0.9426042936271144]
---------
Input: [6.1 2.9 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.008409333505571862, 0.9838160186605163, 0.01596223115415022]
---------
Input: [4.7 3.2 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9935070228613998, 0.007019006769812927, 0.005554838373778736]
---------
Input: [7.7 3.8 6.7 2.2] Expected: [0. 0. 1.] Produced: [0.0009767446112876627, 0.057549183009294826, 0.9425965624889192]
---------
Input: [5.  3.4 1.6 0.4] Expected: [1. 0. 0.] Produced: [0.9933872462727309, 0.007118074273322188, 0.005152580731988836]
---------
Input: [6.7 3.1 4.7 1.5] Expected: [0. 1. 0.] Produced: [0.008851618691793165, 0.9861006832476531, 0.013656092602438163]
---------
Input: [5.8 2.7 4.1 1. ] Expected: [0. 1. 0.] Produced: [0.009018167458165211, 0.9858611002247544, 0.013652482347025005]
---------
Input: [6.5 3.  5.2 2. ] Expected: [0. 0. 1.] Produced: [0.00098380220573782, 0.05868484300522444, 0.9414366968670396]
---------
Input: [5.4 3.9 1.3 0.4] Expected: [1. 0. 0.] Produced: [0.9974484343665616, 0.0029976385754506628, 0.010828681925472958]
---------
Input: [5.4 3.4 1.7 0.2] Expected: [1. 0. 0.] Produced: [0.9948554634634186, 0.005669139230351241, 0.0063360998955078966]
---------
Input: [4.6 3.2 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9956353074121794, 0.0048854933696543465, 0.007291981881034369]
---------
Input: [6.4 3.1 5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0009783750743272265, 0.057738313183145505, 0.9423956345620561]
---------
Input: [7.7 2.6 6.9 2.3] Expected: [0. 0. 1.] Produced: [0.0009765759309510473, 0.05747693473761317, 0.9426694755916801]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9953786855069563, 0.005148988068964575, 0.007094341284922893]
---------
Input: [6.4 3.2 5.3 2.3] Expected: [0. 0. 1.] Produced: [0.0009769783686548997, 0.057457395396277824, 0.9426716412911145]
---------
Input: [6.6 3.  4.4 1.4] Expected: [0. 1. 0.] Produced: [0.008927932579022817, 0.9859842155554028, 0.013631823743093964]
---------
Input: [4.8 3.1 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9928947611465364, 0.007620363371779814, 0.0052507012975597425]
---------
Input: [5.5 2.4 3.8 1.1] Expected: [0. 1. 0.] Produced: [0.009039291251872706, 0.9858115104073288, 0.013658500318245169]
---------
Input: [4.9 2.4 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.009751959582383552, 0.9847660528540797, 0.013430972923270942]
---------
Input: [5.2 3.5 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9965181606034725, 0.003975167290490263, 0.008508907786469564]
---------
Input: [5.  3.  1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9924932450754346, 0.008012157973439554, 0.005055656707673205]
---------
Input: [5.4 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9970595499784506, 0.0034095559600892233, 0.009697279765138195]
---------
Input: [7.2 3.6 6.1 2.5] Expected: [0. 0. 1.] Produced: [0.0009766400804255618, 0.0574434336778171, 0.9426995226520097]
---------
Input: [6.7 3.3 5.7 2.5] Expected: [0. 0. 1.] Produced: [0.000976665944622526, 0.057422991091224555, 0.9427180315816688]
---------
Input: [5.  2.  3.5 1. ] Expected: [0. 1. 0.] Produced: [0.009038832603749889, 0.9856137919116078, 0.013844406945026377]
---------
Input: [4.9 3.  1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9951208405180128, 0.0054062637484660146, 0.006728821685001877]
---------
Input: [6.1 2.8 4.7 1.2] Expected: [0. 1. 0.] Produced: [0.008755478593927012, 0.9857172437094985, 0.014081261053000558]
---------
Input: [5.1 3.8 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9966929051260107, 0.0037931087497866116, 0.00883979710307459]
---------
Input: [5.8 2.7 5.1 1.9] Expected: [0. 0. 1.] Produced: [0.0009769202085651604, 0.05739532063764428, 0.9427372153152253]
---------
Input: [6.4 2.7 5.3 1.9] Expected: [0. 0. 1.] Produced: [0.000977020236542585, 0.05741537923897642, 0.9427202324014933]
---------
Input: [5.1 2.5 3.  1.1] Expected: [0. 1. 0.] Produced: [0.012005114025597221, 0.9814171015795358, 0.012280038671276392]
---------
Input: [5.1 3.5 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9969039937647554, 0.0035730699381646583, 0.009341955937735758]
---------
Input: [5.6 3.  4.1 1.3] Expected: [0. 1. 0.] Produced: [0.009007742124206347, 0.9858423919204564, 0.013636288623218733]
---------
Input: [6.  3.4 4.5 1.6] Expected: [0. 1. 0.] Produced: [0.008882548232852165, 0.9859430855722203, 0.013718558043114849]
---------
Input: [6.4 3.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.00890048212776241, 0.9860140466214365, 0.013647777405289746]
---------
Input: [6.  3.  4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0011129532263631995, 0.08416961778353131, 0.9158471329698031]
---------
Input: [5.1 3.3 1.7 0.5] Expected: [1. 0. 0.] Produced: [0.9883611561562541, 0.011904714683877611, 0.003421002787605586]
---------
Input: [5.2 2.7 3.9 1.4] Expected: [0. 1. 0.] Produced: [0.008580591560914941, 0.983853335660912, 0.015667023269947847]
---------
Input: [4.3 3.  1.1 0.1] Expected: [1. 0. 0.] Produced: [0.9969630110736043, 0.003514230315915367, 0.00968895259542643]
---------
Input: [5.  2.3 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.009645624311892384, 0.9849107314321047, 0.0134876879977584]
---------
Input: [6.4 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.008950550127578203, 0.9859389426562228, 0.013644628582505432]
---------
Input: [7.7 3.  6.1 2.3] Expected: [0. 0. 1.] Produced: [0.0009766826308547979, 0.05736590366166135, 0.9427785006736864]
---------
Input: [7.3 2.9 6.3 1.8] Expected: [0. 0. 1.] Produced: [0.0009766136751173918, 0.05733719914868868, 0.9428078690593554]
---------
Input: [5.5 3.5 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.99739579103211, 0.003053943700459417, 0.010717055403589217]
---------
Input: [6.6 2.9 4.6 1.3] Expected: [0. 1. 0.] Produced: [0.00886415811374829, 0.9860618023298949, 0.013682503803284478]
---------
Input: [7.  3.2 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.00888791208274047, 0.9860277141888798, 0.013660723356149972]
---------
Input: [6.3 2.5 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.0014087507561749148, 0.16391820413662261, 0.836036728544791]
---------
Input: [6.3 3.3 6.  2.5] Expected: [0. 0. 1.] Produced: [0.0009765542493419515, 0.05788490579168591, 0.9422609656570854]
---------
Input: [6.2 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.008933017409961514, 0.9860374503666598, 0.013584203353288544]
---------
Input: [5.6 2.5 3.9 1.1] Expected: [0. 1. 0.] Produced: [0.009020777949429707, 0.9859070733119811, 0.013594867796051147]
---------
Input: [5.9 3.2 4.8 1.8] Expected: [0. 1. 0.] Produced: [0.0015893203687684196, 0.22474422855801093, 0.774794399880108]
---------
Input: [5.  3.5 1.6 0.6] Expected: [1. 0. 0.] Produced: [0.9914124237747356, 0.009046020787745464, 0.0040697355383279565]
---------
Input: [6.7 3.1 4.4 1.4] Expected: [0. 1. 0.] Produced: [0.008962841197981592, 0.9860861248696555, 0.013460796626719954]
---------
Input: [6.8 3.  5.5 2.1] Expected: [0. 0. 1.] Produced: [0.0009825509673425739, 0.05957591936682878, 0.9405626017815257]
---------
Input: [5.  3.2 1.2 0.2] Expected: [1. 0. 0.] Produced: [0.9970372431821031, 0.003438649052992646, 0.00969392661908783]
---------
Input: [6.1 3.  4.9 1.8] Expected: [0. 0. 1.] Produced: [0.0016632132021046223, 0.2538645820481606, 0.7457808238654876]
---------
Input: [4.7 3.2 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9964029239471088, 0.004101940997647805, 0.008375688733555464]
---------
Input: [5.4 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.004048691067023983, 0.8530168153285085, 0.14566060693776814]
---------
Input: [6.2 2.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.006438207291549063, 0.9631357310905062, 0.03649102826803591]
---------
Input: [5.  3.3 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.996363242773008, 0.004142817331712128, 0.008257549209911215]
---------
Input: [6.3 2.7 4.9 1.8] Expected: [0. 0. 1.] Produced: [0.001270756529583298, 0.12549996126634408, 0.8745316142770977]
---------
Input: [5.  3.6 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9970374932213762, 0.0034370092100470978, 0.009663366352546363]
---------
Input: [5.7 3.8 1.7 0.3] Expected: [1. 0. 0.] Produced: [0.9961846135843112, 0.004322927820896965, 0.007771395964683559]
---------
Input: [5.9 3.  4.2 1.5] Expected: [0. 1. 0.] Produced: [0.008935549757686357, 0.9860865068915139, 0.013513635716829895]
---------
Input: [5.6 2.7 4.2 1.3] Expected: [0. 1. 0.] Produced: [0.008863193916659282, 0.9860303653908968, 0.013678726074357082]
---------
Input: [5.8 2.7 3.9 1.2] Expected: [0. 1. 0.] Produced: [0.009105182826328098, 0.9858467234996774, 0.013472015176675463]
---------
Input: [6.4 2.8 5.6 2.1] Expected: [0. 0. 1.] Produced: [0.0009765985115199936, 0.05827754435076813, 0.941868032742205]
---------
Input: [7.4 2.8 6.1 1.9] Expected: [0. 0. 1.] Produced: [0.0009778788097658654, 0.05850086633498847, 0.9416465942007901]
---------
Input: [4.8 3.  1.4 0.3] Expected: [1. 0. 0.] Produced: [0.9940561034656076, 0.00648227032752596, 0.00575895395522511]
---------
Input: [5.7 2.5 5.  2. ] Expected: [0. 0. 1.] Produced: [0.0009767815188262672, 0.05823247002058545, 0.9419056002973896]
---------
Input: [7.9 3.8 6.4 2. ] Expected: [0. 0. 1.] Produced: [0.004625228109905186, 0.9007504230521621, 0.09862539638440902]
---------
Input: [6.  2.2 5.  1.5] Expected: [0. 0. 1.] Produced: [0.0009778068953933674, 0.058069830002911736, 0.9420636168615658]
---------
Input: [6.5 3.  5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0009797526758060298, 0.05844258079262873, 0.941692727230961]
---------
Input: [7.2 3.2 6.  1.8] Expected: [0. 0. 1.] Produced: [0.0009971411973778367, 0.06164310952822583, 0.9384995424828229]
---------
Input: [4.4 2.9 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9935512647262525, 0.006988010003894087, 0.005709974903629547]
---------
Input: [6.7 2.5 5.8 1.8] Expected: [0. 0. 1.] Produced: [0.0009765720119490047, 0.05788741935215355, 0.9422570278806114]
---------
Input: [5.2 3.4 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9967539984648383, 0.003732075537634852, 0.008993828350926858]
---------
Input: [5.5 2.4 3.7 1. ] Expected: [0. 1. 0.] Produced: [0.009180341911744678, 0.9856658933062816, 0.013574164977749305]
---------
Input: [5.1 3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9961514323518013, 0.004357044959559783, 0.007893534472866677]
---------
Input: [4.8 3.4 1.9 0.2] Expected: [1. 0. 0.] Produced: [0.9892430043026479, 0.011151345114275249, 0.004051017318788917]
---------
Input: [5.  3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9960606443135208, 0.004450732971976471, 0.007767644577757389]
---------
Input: [6.9 3.1 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.00882841708718972, 0.9861772491893683, 0.01362084528480269]
---------
Input: [6.8 2.8 4.8 1.4] Expected: [0. 1. 0.] Produced: [0.008823510786342883, 0.9861749269075267, 0.01363298299834071]
---------
Input: [6.9 3.1 5.1 2.3] Expected: [0. 0. 1.] Produced: [0.0010873750838422778, 0.07958345229725086, 0.9205087937059728]
---------
Input: [6.3 3.4 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0009766351895879402, 0.05783579915202176, 0.942302439639559]
---------
Input: [6.2 3.4 5.4 2.3] Expected: [0. 0. 1.] Produced: [0.0009768116215570202, 0.05781578557784517, 0.9423139966646112]
---------
Epoch 5000 RMSE = 0.12318491138314627
Epoch 5100 RMSE = 0.1302519085372812
Epoch 5200 RMSE = 0.13415474494356114
Epoch 5300 RMSE = 0.12376698819748287
Epoch 5400 RMSE = 0.13190443552214165
Epoch 5500 RMSE = 0.12586745511153696
Epoch 5600 RMSE = 0.13322673533561702
Epoch 5700 RMSE = 0.12208878590837484
Epoch 5800 RMSE = 0.130186460347759
Epoch 5900 RMSE = 0.11596406744716074
Input: [6.9 3.1 5.1 2.3] Expected: [0. 0. 1.] Produced: [0.0009300261281853408, 0.06586213222525693, 0.9342109060878511]
---------
Input: [6.3 2.8 5.1 1.5] Expected: [0. 0. 1.] Produced: [0.0011627776803097008, 0.12622581461852908, 0.8738062582161916]
---------
Input: [5.  2.  3.5 1. ] Expected: [0. 1. 0.] Produced: [0.008192682152225874, 0.985899545144527, 0.013633146102047048]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.995753947680615, 0.00466686462150297, 0.006677720620028057]
---------
Input: [6.7 3.3 5.7 2.1] Expected: [0. 0. 1.] Produced: [0.0008829860816516232, 0.05631262931152786, 0.9437918752438103]
---------
Input: [6.8 3.  5.5 2.1] Expected: [0. 0. 1.] Produced: [0.0008832179155631099, 0.05633512220004934, 0.9437678987986339]
---------
Input: [5.4 3.9 1.3 0.4] Expected: [1. 0. 0.] Produced: [0.9976488551782986, 0.002745693282427346, 0.010095279952550051]
---------
Input: [6.2 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.008096195496269493, 0.9861890156645774, 0.01349577473779436]
---------
Input: [4.8 3.1 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9934394926730772, 0.006897632501360079, 0.004950914437136825]
---------
Input: [6.4 3.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.008068603570277645, 0.986231746482352, 0.013492990576577829]
---------
Input: [4.4 2.9 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9940481993807281, 0.006325512671369983, 0.005382790382315341]
---------
Input: [5.9 3.2 4.8 1.8] Expected: [0. 1. 0.] Produced: [0.00106862290871616, 0.09855586088388492, 0.9013643135103278]
---------
Input: [7.4 2.8 6.1 1.9] Expected: [0. 0. 1.] Produced: [0.00088289637042752, 0.056699378190766266, 0.9434112848377719]
---------
Input: [6.1 2.8 4.7 1.2] Expected: [0. 1. 0.] Produced: [0.007966091175924532, 0.9861262056194211, 0.013729677703577905]
---------
Input: [5.  3.2 1.2 0.2] Expected: [1. 0. 0.] Produced: [0.9972764268620171, 0.003133832113125395, 0.009079327429141906]
---------
Input: [6.3 2.5 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.0012992854940209948, 0.1718166878508821, 0.8281460451965393]
---------
Input: [6.5 2.8 4.6 1.5] Expected: [0. 1. 0.] Produced: [0.008010361166869494, 0.9864310761483162, 0.013413067885248056]
---------
Input: [5.8 2.7 5.1 1.9] Expected: [0. 0. 1.] Produced: [0.0008830283657404138, 0.05722966633386182, 0.9428710102153561]
---------
Input: [5.4 3.4 1.7 0.2] Expected: [1. 0. 0.] Produced: [0.9952846824083875, 0.005128068195534789, 0.005977568699930197]
---------
Input: [5.6 2.9 3.6 1.3] Expected: [0. 1. 0.] Produced: [0.008769448403715826, 0.9852420254258726, 0.01296429554484388]
---------
Input: [4.6 3.2 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9959847524484219, 0.004443138953175132, 0.006841539208490739]
---------
Input: [6.7 2.5 5.8 1.8] Expected: [0. 0. 1.] Produced: [0.0008827978685542278, 0.05722522671447944, 0.9428861905096141]
---------
Input: [5.1 3.3 1.7 0.5] Expected: [1. 0. 0.] Produced: [0.9891842775342512, 0.01080029741498266, 0.00322189161239613]
---------
Input: [5.6 2.5 3.9 1.1] Expected: [0. 1. 0.] Produced: [0.008171182280829232, 0.986199207942357, 0.013375056900584804]
---------
Input: [4.9 3.  1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9955261435510739, 0.004896623582094804, 0.006338516524061955]
---------
Input: [6.  2.2 5.  1.5] Expected: [0. 0. 1.] Produced: [0.0008837727245762017, 0.05732111249777901, 0.9427796599010689]
---------
Input: [7.7 2.6 6.9 2.3] Expected: [0. 0. 1.] Produced: [0.0008827403024891857, 0.05719456736640669, 0.9429179742518945]
---------
Input: [5.6 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.006354928866666191, 0.9716522721681411, 0.02804076567003135]
---------
Input: [7.2 3.2 6.  1.8] Expected: [0. 0. 1.] Produced: [0.0009041905525681426, 0.061457581720858454, 0.9386524082930321]
---------
Input: [6.7 3.3 5.7 2.5] Expected: [0. 0. 1.] Produced: [0.000882817015264669, 0.057157149306532574, 0.9429507486238478]
---------
Input: [5.2 2.7 3.9 1.4] Expected: [0. 1. 0.] Produced: [0.008007774443688987, 0.9856637193227127, 0.013969549906145745]
---------
Input: [5.6 2.7 4.2 1.3] Expected: [0. 1. 0.] Produced: [0.008029114956062814, 0.9862064561820909, 0.013561685019567591]
---------
Input: [7.9 3.8 6.4 2. ] Expected: [0. 0. 1.] Produced: [0.002552658953850989, 0.6484635697879692, 0.3509656614708653]
---------
Input: [5.7 4.4 1.5 0.4] Expected: [1. 0. 0.] Produced: [0.9978129194572275, 0.0025745683587783726, 0.010707068600601787]
---------
Input: [6.2 2.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.0015374075827242365, 0.2617061939953935, 0.7379511725039642]
---------
Input: [5.  2.3 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.008713024117450332, 0.9853283752136206, 0.013201694370982248]
---------
Input: [5.4 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9972951370135686, 0.003115132594887458, 0.009060772190940463]
---------
Input: [5.2 3.4 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9970201314678663, 0.003397701674742054, 0.008432335506784919]
---------
Input: [6.4 3.1 5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0008836356431139032, 0.057326957701529464, 0.9427738780595488]
---------
Input: [4.5 2.3 1.3 0.3] Expected: [1. 0. 0.] Produced: [0.9868052309195596, 0.013060893870221784, 0.0035173811359260985]
---------
Input: [6.1 2.9 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.007660777101606259, 0.9843912216487333, 0.015451269148456034]
---------
Input: [7.3 2.9 6.3 1.8] Expected: [0. 0. 1.] Produced: [0.0008827744749277094, 0.05719865781964276, 0.9429125924998797]
---------
Input: [5.5 3.5 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9976006494647185, 0.0027987926241761028, 0.009983990921485457]
---------
Input: [6.4 2.8 5.6 2.1] Expected: [0. 0. 1.] Produced: [0.0008828042159080579, 0.05717673503933387, 0.9429321839584699]
---------
Input: [7.7 3.  6.1 2.3] Expected: [0. 0. 1.] Produced: [0.0008828105217979991, 0.05717200630156786, 0.9429383245315721]
---------
Input: [6.3 3.4 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.000882844276447492, 0.057141171684955916, 0.9429639556047548]
---------
Input: [5.7 3.8 1.7 0.3] Expected: [1. 0. 0.] Produced: [0.9965128883633565, 0.003907123716835831, 0.007340097818060526]
---------
Input: [6.2 3.4 5.4 2.3] Expected: [0. 0. 1.] Produced: [0.0008829686190764441, 0.057118874697119126, 0.9429785412723051]
---------
Input: [6.9 3.2 5.7 2.3] Expected: [0. 0. 1.] Produced: [0.0008828487872284129, 0.057112906880292164, 0.9429932356333482]
---------
Input: [5.5 2.4 3.8 1.1] Expected: [0. 1. 0.] Produced: [0.008186839491245417, 0.9861441924165301, 0.013401155193178577]
---------
Input: [6.  3.4 4.5 1.6] Expected: [0. 1. 0.] Produced: [0.008056087669730967, 0.9862865118048012, 0.013440704972388048]
---------
Input: [6.9 3.1 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.008007284155492872, 0.9864274492207543, 0.013425586519900562]
---------
Input: [4.6 3.1 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9944857960877315, 0.00591059845413992, 0.005547596207563186]
---------
Input: [6.4 3.2 5.3 2.3] Expected: [0. 0. 1.] Produced: [0.000883008829706691, 0.05709137397072991, 0.9430044390157235]
---------
Input: [5.1 3.8 1.5 0.3] Expected: [1. 0. 0.] Produced: [0.9970399186150226, 0.003375409671666357, 0.008401632194521853]
---------
Input: [5.6 3.  4.1 1.3] Expected: [0. 1. 0.] Produced: [0.008160153952272554, 0.9861787106652445, 0.013375764166649992]
---------
Input: [4.8 3.4 1.9 0.2] Expected: [1. 0. 0.] Produced: [0.9899347450174147, 0.010174382544728055, 0.0037940644540234657]
---------
Input: [6.5 3.  5.2 2. ] Expected: [0. 0. 1.] Produced: [0.000886865909878803, 0.05779452915982539, 0.9422937460618777]
---------
Input: [5.  3.4 1.6 0.4] Expected: [1. 0. 0.] Produced: [0.9939234617687068, 0.006428612809057976, 0.004869793329117223]
---------
Input: [6.  3.  4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0009739488786145312, 0.07619563562150343, 0.923810018462942]
---------
Input: [5.1 3.8 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9969576825446116, 0.003460001720642856, 0.0082717431885849]
---------
Input: [6.7 3.1 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0008828081694836718, 0.05703337559089237, 0.943072889370413]
---------
Input: [6.6 3.  4.4 1.4] Expected: [0. 1. 0.] Produced: [0.008091466036451738, 0.9862895748499639, 0.01339339481077725]
---------
Input: [7.2 3.6 6.1 2.5] Expected: [0. 0. 1.] Produced: [0.0008827693017303463, 0.05702009482398721, 0.9430880919331897]
---------
Input: [5.7 2.5 5.  2. ] Expected: [0. 0. 1.] Produced: [0.0008829358846498454, 0.05699334979292815, 0.9431080627770558]
---------
Input: [6.5 3.2 5.1 2. ] Expected: [0. 0. 1.] Produced: [0.0009768463810199378, 0.07698041908059364, 0.9230706729468693]
---------
Input: [4.3 3.  1.1 0.1] Expected: [1. 0. 0.] Produced: [0.9971974945806509, 0.0032186002202022794, 0.009026846721032539]
---------
Input: [5.  3.3 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9966642390949888, 0.0037583517081005617, 0.007767533033683197]
---------
Input: [5.  3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9963842967448334, 0.0040394981660312615, 0.007303733543541312]
---------
Input: [5.1 2.5 3.  1.1] Expected: [0. 1. 0.] Produced: [0.010791997984346265, 0.9819747556949235, 0.012044669850114471]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9957542051426637, 0.004671993709904049, 0.00666970340137612]
---------
Input: [5.8 2.8 5.1 2.4] Expected: [0. 0. 1.] Produced: [0.0008828596968697395, 0.0569528910333969, 0.9431501160855958]
---------
Input: [4.6 3.6 1.  0.2] Expected: [1. 0. 0.] Produced: [0.9979673312824521, 0.0024135867293366225, 0.011457449905852062]
---------
Input: [5.  3.5 1.6 0.6] Expected: [1. 0. 0.] Produced: [0.9920914974299773, 0.008125047727477483, 0.00386963571991566]
---------
Input: [5.9 3.  4.2 1.5] Expected: [0. 1. 0.] Produced: [0.008095734366972106, 0.9862468712573003, 0.013417353914255158]
---------
Input: [4.7 3.2 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9940065252030416, 0.006367469129769262, 0.005225025516931603]
---------
Input: [6.5 3.  5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0008836768809955846, 0.05707606356477916, 0.9430248002606093]
---------
Input: [6.4 2.7 5.3 1.9] Expected: [0. 0. 1.] Produced: [0.000882999843504053, 0.05693244633623969, 0.9431691538032113]
---------
Input: [6.3 3.3 6.  2.5] Expected: [0. 0. 1.] Produced: [0.0008827341399491443, 0.05691102338903996, 0.9431985924137082]
---------
Input: [5.  3.  1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9930832396323378, 0.0072450556016261285, 0.004771050607736578]
---------
Input: [5.7 2.8 4.5 1.3] Expected: [0. 1. 0.] Produced: [0.007211188554906367, 0.9809097690469991, 0.01887908189161859]
---------
Input: [5.5 2.4 3.7 1. ] Expected: [0. 1. 0.] Produced: [0.008308701436477635, 0.9859305479517689, 0.01339291184057249]
---------
Input: [7.  3.2 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.008057508409174102, 0.9863284350735936, 0.01342255761120693]
---------
Input: [4.9 2.5 4.5 1.7] Expected: [0. 0. 1.] Produced: [0.000883686903404417, 0.056843218054630294, 0.943222218361408]
---------
Input: [5.8 2.7 3.9 1.2] Expected: [0. 1. 0.] Produced: [0.008244143864369141, 0.9860288549896928, 0.01336644494981687]
---------
Input: [6.1 3.  4.9 1.8] Expected: [0. 0. 1.] Produced: [0.0009435166900069639, 0.06919592548059209, 0.930842615560246]
---------
Input: [6.7 3.1 4.4 1.4] Expected: [0. 1. 0.] Produced: [0.008121842552210991, 0.9862202082179038, 0.013392495477122601]
---------
Input: [4.8 3.  1.4 0.3] Expected: [1. 0. 0.] Produced: [0.9945489431563097, 0.005838945813290774, 0.005455337919785605]
---------
Input: [7.7 3.8 6.7 2.2] Expected: [0. 0. 1.] Produced: [0.0008827638517506534, 0.05686648066123909, 0.9432430850127973]
---------
Input: [6.3 2.7 4.9 1.8] Expected: [0. 0. 1.] Produced: [0.0009159716334803939, 0.06335949906758422, 0.9367092706109139]
---------
Input: [5.9 3.  5.1 1.8] Expected: [0. 0. 1.] Produced: [0.0008836526355493359, 0.056873179388910865, 0.9432102153333792]
---------
Input: [6.4 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.008110801709954138, 0.9862320444744599, 0.013419843254816417]
---------
Input: [5.8 2.7 4.1 1. ] Expected: [0. 1. 0.] Produced: [0.008167578726002898, 0.9861434887816966, 0.013441542519055343]
---------
Input: [6.2 2.8 4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0010009860340093166, 0.08237983447530166, 0.9176440698174456]
---------
Input: [5.4 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.0015477297108227848, 0.2646612876967703, 0.7344519059591946]
---------
Input: [4.7 3.2 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.996694126683349, 0.0037322166286009612, 0.007855117040990775]
---------
Input: [5.1 3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9964694044813266, 0.0039573706317904914, 0.007416621900067798]
---------
Input: [6.6 2.9 4.6 1.3] Expected: [0. 1. 0.] Produced: [0.008037042397885592, 0.9864419971113391, 0.013362290423376343]
---------
Input: [4.9 2.4 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.00880467386279345, 0.9852274335174607, 0.013115390154214757]
---------
Input: [5.3 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9972491348303854, 0.003163679716572583, 0.008941861918252811]
---------
Input: [5.  3.6 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9972715893044427, 0.0031410202223087448, 0.009025782098939889]
---------
Input: [6.8 2.8 4.8 1.4] Expected: [0. 1. 0.] Produced: [0.008004291733227803, 0.98648947351517, 0.013373173642290275]
---------
Input: [5.2 3.5 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9968034216592093, 0.0036195032979002098, 0.007975154612368343]
---------
Input: [6.7 3.1 4.7 1.5] Expected: [0. 1. 0.] Produced: [0.008026517880014275, 0.9864579331434608, 0.013357715501381885]
---------
Input: [5.1 3.5 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9971525849538356, 0.0032634158429265024, 0.008729912633070061]
---------
Epoch 6000 RMSE = 0.1414818470640784
Epoch 6100 RMSE = 0.13749558390288646
Epoch 6200 RMSE = 0.13649375479085296
Epoch 6300 RMSE = 0.13937430974140216
Epoch 6400 RMSE = 0.1384018398510845
Epoch 6500 RMSE = 0.14258825182542373
Epoch 6600 RMSE = 0.1292716566962795
Epoch 6700 RMSE = 0.1255497644476753
Epoch 6800 RMSE = 0.13606970252650363
Epoch 6900 RMSE = 0.12060316031867827
Input: [6.5 2.8 4.6 1.5] Expected: [0. 1. 0.] Produced: [0.0073829247388645936, 0.9861030087476913, 0.013785460025901926]
---------
Input: [4.9 2.5 4.5 1.7] Expected: [0. 0. 1.] Produced: [0.0008120097366444577, 0.053492861807642914, 0.9465457338971028]
---------
Input: [4.8 3.  1.4 0.3] Expected: [1. 0. 0.] Produced: [0.9949581384624744, 0.005302651583720044, 0.005240624835244117]
---------
Input: [5.2 2.7 3.9 1.4] Expected: [0. 1. 0.] Produced: [0.00743265332818273, 0.985697070614802, 0.01399393999435431]
---------
Input: [4.6 3.2 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9962776737451835, 0.004061316053663654, 0.006526969099479886]
---------
Input: [4.6 3.1 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9948820874314831, 0.00538163508497253, 0.00531576852219431]
---------
Input: [5.4 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9974885680362636, 0.0028717434642735475, 0.008571651606657911]
---------
Input: [6.4 2.8 5.6 2.1] Expected: [0. 0. 1.] Produced: [0.0008112523709875306, 0.053521525784779464, 0.9465541668849906]
---------
Input: [4.9 3.  1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9958640076278069, 0.0044565267133639735, 0.006073416507927195]
---------
Input: [5.7 2.5 5.  2. ] Expected: [0. 0. 1.] Produced: [0.0008113701118523818, 0.05350101758510198, 0.9465689262772409]
---------
Input: [5.6 2.7 4.2 1.3] Expected: [0. 1. 0.] Produced: [0.007411747573282538, 0.9859704035060319, 0.013845307963658415]
---------
Input: [6.3 3.4 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0008112840253523852, 0.053492627171748935, 0.9465798364890796]
---------
Input: [7.7 3.  6.1 2.3] Expected: [0. 0. 1.] Produced: [0.0008114173427772638, 0.053523254926164665, 0.946553575233121]
---------
Input: [5.5 2.4 3.8 1.1] Expected: [0. 1. 0.] Produced: [0.0075354756580838355, 0.9858373214451538, 0.013757849898270804]
---------
Input: [6.  2.2 5.  1.5] Expected: [0. 0. 1.] Produced: [0.000812030932132905, 0.053564002755253214, 0.946503801933857]
---------
Input: [5.  2.3 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.008001212078386994, 0.9850193439530154, 0.01356637053270459]
---------
Input: [6.7 3.3 5.7 2.5] Expected: [0. 0. 1.] Produced: [0.0008112486536333408, 0.05345443396307606, 0.946619848892506]
---------
Input: [5.  3.5 1.6 0.6] Expected: [1. 0. 0.] Produced: [0.9926790262849002, 0.007327436221266719, 0.003746268534506322]
---------
Input: [5.6 2.5 3.9 1.1] Expected: [0. 1. 0.] Produced: [0.0075210170483081895, 0.9858592533748655, 0.013763450870566785]
---------
Input: [5.6 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.006593239680609513, 0.9799887665872828, 0.019848327656359483]
---------
Input: [5.4 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.003522477762754919, 0.8652102486342433, 0.13407633881297995]
---------
Input: [5.2 3.5 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9970399946115908, 0.0033172796048826754, 0.007588967075594799]
---------
Input: [7.  3.2 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.0074219240993741616, 0.9860425480758256, 0.01375612468330092]
---------
Input: [6.7 3.1 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0008112538985279644, 0.05349601677240186, 0.9465784825249636]
---------
Input: [4.4 2.9 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9944685533716938, 0.005770917005801018, 0.005147717645843932]
---------
Input: [5.8 2.7 3.9 1.2] Expected: [0. 1. 0.] Produced: [0.007585554030675391, 0.9857536800317676, 0.01369532203360182]
---------
Input: [5.1 2.5 3.  1.1] Expected: [0. 1. 0.] Produced: [0.009860600587479406, 0.9817480381653306, 0.012346239999153442]
---------
Input: [6.9 3.2 5.7 2.3] Expected: [0. 0. 1.] Produced: [0.0008113536088151427, 0.05350241101905489, 0.9465719818581929]
---------
Input: [5.  3.6 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9974632780935513, 0.0028975851782736847, 0.00853315055807313]
---------
Input: [4.9 2.4 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.008082528601778613, 0.9848824806455708, 0.013513443715844417]
---------
Input: [6.4 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.007468382950880618, 0.9859578990467298, 0.0137440472899955]
---------
Input: [5.  3.  1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9935857578834799, 0.006570101575909576, 0.004592616485289876]
---------
Input: [7.9 3.8 6.4 2. ] Expected: [0. 0. 1.] Produced: [0.0038777778749697236, 0.8991402803253175, 0.10063896671674653]
---------
Input: [5.  2.  3.5 1. ] Expected: [0. 1. 0.] Produced: [0.00755009076535385, 0.9857065972997566, 0.013879393505701046]
---------
Input: [5.  3.3 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.996909871329654, 0.0034439351727851048, 0.007392082702264359]
---------
Input: [6.3 2.8 5.1 1.5] Expected: [0. 0. 1.] Produced: [0.001286367809204537, 0.1991707885230887, 0.800797099328369]
---------
Input: [6.7 2.5 5.8 1.8] Expected: [0. 0. 1.] Produced: [0.0008112073829911855, 0.05306342139169487, 0.9470134441285228]
---------
Input: [6.2 3.4 5.4 2.3] Expected: [0. 0. 1.] Produced: [0.0008113624191811283, 0.05304240335509878, 0.9470234193286434]
---------
Input: [5.4 3.9 1.3 0.4] Expected: [1. 0. 0.] Produced: [0.997810374456676, 0.002545051308081767, 0.009499090726133688]
---------
Input: [5.7 4.4 1.5 0.4] Expected: [1. 0. 0.] Produced: [0.9979571715584747, 0.002395493884231444, 0.01003773826461501]
---------
Input: [4.8 3.4 1.9 0.2] Expected: [1. 0. 0.] Produced: [0.990557868733927, 0.009241850899568632, 0.003652596571635802]
---------
Input: [7.2 3.2 6.  1.8] Expected: [0. 0. 1.] Produced: [0.0008175007025619392, 0.05429358984170213, 0.9457813214379435]
---------
Input: [5.7 3.8 1.7 0.3] Expected: [1. 0. 0.] Produced: [0.9967836954812359, 0.0035607783970755037, 0.007024380900688205]
---------
Input: [5.1 3.8 1.5 0.3] Expected: [1. 0. 0.] Produced: [0.997254264761719, 0.0031017822495979274, 0.007973651866362341]
---------
Input: [5.1 3.3 1.7 0.5] Expected: [1. 0. 0.] Produced: [0.989934969474, 0.009706734471495013, 0.003132553295831536]
---------
Input: [7.7 2.6 6.9 2.3] Expected: [0. 0. 1.] Produced: [0.0008111675348375924, 0.053024724457157085, 0.9470529360457618]
---------
Input: [6.9 3.1 5.1 2.3] Expected: [0. 0. 1.] Produced: [0.0008815307617693289, 0.0681053026034378, 0.9319385975783037]
---------
Input: [6.9 3.1 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.007378641444273474, 0.9860256056444852, 0.013871252337294866]
---------
Input: [6.4 2.7 5.3 1.9] Expected: [0. 0. 1.] Produced: [0.0008114063677815337, 0.05299487855565142, 0.9470747438659407]
---------
Input: [5.  3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9966525518176772, 0.0036926432892010995, 0.0069676259223907105]
---------
Input: [6.7 3.1 4.4 1.4] Expected: [0. 1. 0.] Produced: [0.007478110405941881, 0.985848968431027, 0.013812258120834002]
---------
Input: [7.3 2.9 6.3 1.8] Expected: [0. 0. 1.] Produced: [0.0008111975649157886, 0.05297837544691823, 0.9470984318606798]
---------
Input: [6.5 3.2 5.1 2. ] Expected: [0. 0. 1.] Produced: [0.0009610843564492224, 0.08796836374143992, 0.9120531638266651]
---------
Input: [6.7 3.3 5.7 2.1] Expected: [0. 0. 1.] Produced: [0.000811371652417105, 0.05294223300070136, 0.9471289131702777]
---------
Input: [6.8 3.  5.5 2.1] Expected: [0. 0. 1.] Produced: [0.000811609922275764, 0.05297094743777592, 0.9470989270987039]
---------
Input: [6.1 3.  4.9 1.8] Expected: [0. 0. 1.] Produced: [0.0008970840104397266, 0.07148745956099611, 0.928524875498295]
---------
Input: [6.3 2.5 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.0011893062556139742, 0.1608988693406872, 0.8390899165049025]
---------
Input: [5.8 2.7 4.1 1. ] Expected: [0. 1. 0.] Produced: [0.007517774131756683, 0.9858479152162327, 0.013787207338658536]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9960677879990953, 0.0042645338924972645, 0.006372087418194529]
---------
Input: [4.7 3.2 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9969350024494061, 0.003421541389946719, 0.0074714719503541525]
---------
Input: [4.7 3.2 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9944325906851125, 0.005792672023525193, 0.005012013777161729]
---------
Input: [5.3 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9974460716314227, 0.0029136971095661958, 0.008467392285251398]
---------
Input: [6.1 2.9 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.007306846605127311, 0.9856528649924606, 0.014248725903281896]
---------
Input: [6.3 2.7 4.9 1.8] Expected: [0. 0. 1.] Produced: [0.0009055783765520033, 0.07430329793285956, 0.9257358844536893]
---------
Input: [5.9 3.  5.1 1.8] Expected: [0. 0. 1.] Produced: [0.0008126033049584774, 0.05355279716603766, 0.9465014342348679]
---------
Input: [6.6 3.  4.4 1.4] Expected: [0. 1. 0.] Produced: [0.007451282730551793, 0.9859567305593628, 0.013774324191170266]
---------
Input: [6.2 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.0074550745499460625, 0.9859507834793793, 0.01378369934222418]
---------
Input: [5.5 3.5 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9977666507724721, 0.0025911646294759783, 0.009404637915655824]
---------
Input: [6.2 2.8 4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0011854015550616292, 0.16051226812648936, 0.839414576482272]
---------
Input: [6.7 3.1 4.7 1.5] Expected: [0. 1. 0.] Produced: [0.0073950035795018855, 0.9860411283273468, 0.013818908223768992]
---------
Input: [5.9 3.2 4.8 1.8] Expected: [0. 1. 0.] Produced: [0.0013941823994105497, 0.24326820086790596, 0.7563885333950621]
---------
Input: [6.4 3.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.007431097580959104, 0.9860734229998152, 0.013701593727657488]
---------
Input: [4.8 3.1 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9939078676874035, 0.0062834918906242345, 0.00474292988798829]
---------
Input: [4.6 3.6 1.  0.2] Expected: [1. 0. 0.] Produced: [0.9980904857021795, 0.0022607748283327567, 0.010669344224838652]
---------
Input: [5.7 2.8 4.5 1.3] Expected: [0. 1. 0.] Produced: [0.00732181095391376, 0.985742183868538, 0.014142826139585066]
---------
Input: [5.1 3.5 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9973571130924382, 0.003004856568176484, 0.00826818421791151]
---------
Input: [6.5 3.  5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0008176855583537294, 0.05517294020867516, 0.9448970131820783]
---------
Input: [6.1 2.8 4.7 1.2] Expected: [0. 1. 0.] Produced: [0.0073775975485461396, 0.9861367577329534, 0.013766016731992568]
---------
Input: [7.4 2.8 6.1 1.9] Expected: [0. 0. 1.] Produced: [0.0008117826401666736, 0.053992636839633845, 0.9460844521614895]
---------
Input: [5.8 2.8 5.1 2.4] Expected: [0. 0. 1.] Produced: [0.0008112658763276896, 0.053850695267419, 0.946220358570859]
---------
Input: [5.2 3.4 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9972386454355209, 0.0031225856473458226, 0.008000470442192086]
---------
Input: [5.6 3.  4.1 1.3] Expected: [0. 1. 0.] Produced: [0.0075120521378630595, 0.9859261098345763, 0.013680502102765803]
---------
Input: [6.4 3.2 5.3 2.3] Expected: [0. 0. 1.] Produced: [0.0008116391804087769, 0.05389005237256318, 0.9461743594627963]
---------
Input: [7.7 3.8 6.7 2.2] Expected: [0. 0. 1.] Produced: [0.0008117738652325354, 0.0539534939467873, 0.9461237110133099]
---------
Input: [6.  3.4 4.5 1.6] Expected: [0. 1. 0.] Produced: [0.0074281851343226905, 0.9860660781179488, 0.013710464985020725]
---------
Input: [5.6 2.9 3.6 1.3] Expected: [0. 1. 0.] Produced: [0.008050179209316203, 0.9849704224252137, 0.013300886582032077]
---------
Input: [5.9 3.  4.2 1.5] Expected: [0. 1. 0.] Produced: [0.007457807251154851, 0.9860162793195995, 0.013698493109880727]
---------
Input: [6.  3.  4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0017088226610547097, 0.38573632310728745, 0.6138374244153663]
---------
Input: [6.3 3.3 6.  2.5] Expected: [0. 0. 1.] Produced: [0.000811155078153009, 0.05343092306813116, 0.946646280553877]
---------
Input: [6.2 2.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.004712824515113349, 0.943322395776055, 0.056449244474706874]
---------
Input: [4.3 3.  1.1 0.1] Expected: [1. 0. 0.] Produced: [0.9973885458601063, 0.0029755548692537858, 0.008512995515847365]
---------
Input: [5.8 2.7 5.1 1.9] Expected: [0. 0. 1.] Produced: [0.0008113635933531216, 0.05341575917342255, 0.9466518605485424]
---------
Input: [5.4 3.4 1.7 0.2] Expected: [1. 0. 0.] Produced: [0.9956489226585031, 0.0046522068960613595, 0.005746600606376073]
---------
Input: [6.4 3.1 5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0008131926875044915, 0.053775111688352424, 0.9462931700834588]
---------
Input: [5.1 3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9967332063616441, 0.003616751462984385, 0.007076623494657547]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9960668952601021, 0.004265410691471666, 0.006370522665194578]
---------
Input: [6.6 2.9 4.6 1.3] Expected: [0. 1. 0.] Produced: [0.00740380694064875, 0.9860471570242482, 0.013801896283154742]
---------
Input: [6.5 3.  5.2 2. ] Expected: [0. 0. 1.] Produced: [0.0008235969958422466, 0.055855491374333405, 0.9442029846089331]
---------
Input: [5.5 2.4 3.7 1. ] Expected: [0. 1. 0.] Produced: [0.007641918725795573, 0.9856291801764944, 0.013746958501249911]
---------
Input: [5.  3.2 1.2 0.2] Expected: [1. 0. 0.] Produced: [0.9974703468237809, 0.0028905819622803085, 0.008582453389442433]
---------
Input: [4.5 2.3 1.3 0.3] Expected: [1. 0. 0.] Produced: [0.9876634134895428, 0.011802485766980937, 0.003405525509932888]
---------
Input: [5.1 3.8 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9971754746104057, 0.0031823513588577356, 0.007842698687878444]
---------
Input: [7.2 3.6 6.1 2.5] Expected: [0. 0. 1.] Produced: [0.0008111838876600822, 0.05338615351521315, 0.9466892184028013]
---------
Input: [5.  3.4 1.6 0.4] Expected: [1. 0. 0.] Produced: [0.9943820578726836, 0.005816361563083938, 0.004696338774242235]
---------
Input: [6.8 2.8 4.8 1.4] Expected: [0. 1. 0.] Produced: [0.0073757942758830184, 0.9860901910011418, 0.013815884012224415]
---------
Epoch 7000 RMSE = 0.12428677845144503
Epoch 7100 RMSE = 0.14240079970310135
Epoch 7200 RMSE = 0.1310316520998952
Epoch 7300 RMSE = 0.11500616111586032
Epoch 7400 RMSE = 0.11187699730469554
Epoch 7500 RMSE = 0.14490792792482826
Epoch 7600 RMSE = 0.12064718079927424
Epoch 7700 RMSE = 0.12582810823712165
Epoch 7800 RMSE = 0.09440663339664897
Epoch 7900 RMSE = 0.1338161656629042
Input: [5.8 2.7 4.1 1. ] Expected: [0. 1. 0.] Produced: [0.006999768634512946, 0.9867282584063862, 0.012972469588346492]
---------
Input: [5.4 3.9 1.3 0.4] Expected: [1. 0. 0.] Produced: [0.9979364990440133, 0.002407834395981615, 0.008902028092368515]
---------
Input: [6.4 3.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.006923112942996568, 0.9868615447184522, 0.012965366491267977]
---------
Input: [4.9 2.4 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.007507396727214009, 0.9858288025141415, 0.012735870566221082]
---------
Input: [6.7 2.5 5.8 1.8] Expected: [0. 0. 1.] Produced: [0.0007543361766019303, 0.057104233839452435, 0.9429415370050245]
---------
Input: [7.2 3.6 6.1 2.5] Expected: [0. 0. 1.] Produced: [0.0007543374724365926, 0.05708827853829563, 0.9429562792538884]
---------
Input: [6.5 2.8 4.6 1.5] Expected: [0. 1. 0.] Produced: [0.0068796818990233995, 0.9869226087094297, 0.013007199048853358]
---------
Input: [4.3 3.  1.1 0.1] Expected: [1. 0. 0.] Produced: [0.9975388362116647, 0.002814296103858058, 0.00797795964140053]
---------
Input: [5.9 3.  4.2 1.5] Expected: [0. 1. 0.] Produced: [0.006946604438632703, 0.9868128691415173, 0.01295927026261733]
---------
Input: [5.7 2.8 4.5 1.3] Expected: [0. 1. 0.] Produced: [0.006673443906642129, 0.9855523202465495, 0.014369835398131322]
---------
Input: [5.9 3.2 4.8 1.8] Expected: [0. 1. 0.] Produced: [0.0011371002440274552, 0.18422715725468988, 0.8155471980323268]
---------
Input: [7.7 2.6 6.9 2.3] Expected: [0. 0. 1.] Produced: [0.0007542964252108413, 0.05767644582837159, 0.9423705156926824]
---------
Input: [6.8 3.  5.5 2.1] Expected: [0. 0. 1.] Produced: [0.0007553441220664469, 0.057868534141503884, 0.9421709301409065]
---------
Input: [5.3 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9975971934598081, 0.002753444834688359, 0.007943512485664847]
---------
Input: [5.  3.  1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9939709233459434, 0.006208370539957914, 0.004310939374681858]
---------
Input: [7.9 3.8 6.4 2. ] Expected: [0. 0. 1.] Produced: [0.003408351754392129, 0.8881017332245599, 0.11181764290778932]
---------
Input: [6.  2.2 5.  1.5] Expected: [0. 0. 1.] Produced: [0.0007546586495654091, 0.05733764963982257, 0.942698479437696]
---------
Input: [6.6 2.9 4.6 1.3] Expected: [0. 1. 0.] Produced: [0.006898968733909436, 0.9869178764300517, 0.012976690930245572]
---------
Input: [6.3 2.8 5.1 1.5] Expected: [0. 0. 1.] Produced: [0.001096018756141347, 0.16820123008134594, 0.8317844612794826]
---------
Input: [5.4 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9976378876212448, 0.002710899377286853, 0.00804935640879472]
---------
Input: [5.1 2.5 3.  1.1] Expected: [0. 1. 0.] Produced: [0.009119964216714922, 0.9829526996302508, 0.011645930648803203]
---------
Input: [4.8 3.1 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9942663481684918, 0.005929609670007208, 0.0044581918247254895]
---------
Input: [6.4 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.006955942209548856, 0.9868014212870048, 0.01295953954665796]
---------
Input: [5.5 3.5 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9978953806400483, 0.0024506343286607468, 0.00881589363390477]
---------
Input: [6.4 3.2 5.3 2.3] Expected: [0. 0. 1.] Produced: [0.000754482999863274, 0.05718806087013713, 0.9428451633978966]
---------
Input: [5.  3.6 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.997611978203944, 0.0027374299892665944, 0.008006754171196043]
---------
Input: [7.4 2.8 6.1 1.9] Expected: [0. 0. 1.] Produced: [0.0007543497469156204, 0.057185267612046106, 0.9428602405460561]
---------
Input: [5.7 3.8 1.7 0.3] Expected: [1. 0. 0.] Produced: [0.9969887790504569, 0.0033510543433726745, 0.006617026787611473]
---------
Input: [5.6 2.5 3.9 1.1] Expected: [0. 1. 0.] Produced: [0.0070023875466032484, 0.9867163462036823, 0.012972518956799458]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9963072946806021, 0.004021530595041023, 0.005989778236069034]
---------
Input: [7.7 3.  6.1 2.3] Expected: [0. 0. 1.] Produced: [0.0007543339431004221, 0.057167288322832424, 0.9428779893551038]
---------
Input: [7.2 3.2 6.  1.8] Expected: [0. 0. 1.] Produced: [0.0007583413776271126, 0.05806359872577021, 0.9419796743160183]
---------
Input: [6.5 3.  5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0007548889861272247, 0.05722165009441986, 0.9428162032584589]
---------
Input: [6.1 2.9 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.006752297115354055, 0.9861524525958852, 0.013788622656179337]
---------
Input: [5.8 2.7 3.9 1.2] Expected: [0. 1. 0.] Produced: [0.0070602611133390005, 0.9866064239259652, 0.012923883477987822]
---------
Input: [5.6 2.7 4.2 1.3] Expected: [0. 1. 0.] Produced: [0.006890163982559859, 0.9867116219567714, 0.013151704114318664]
---------
Input: [6.2 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.006944040647225596, 0.9868130391027575, 0.012978034592810082]
---------
Input: [4.6 3.1 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9951859796660109, 0.00507973207990835, 0.00499350827708589]
---------
Input: [4.5 2.3 1.3 0.3] Expected: [1. 0. 0.] Produced: [0.9883139650428544, 0.011216914229516997, 0.0031850218111515597]
---------
Input: [5.5 2.4 3.7 1. ] Expected: [0. 1. 0.] Produced: [0.007111008868853739, 0.9865195420993412, 0.012944568413709908]
---------
Input: [6.5 3.2 5.1 2. ] Expected: [0. 0. 1.] Produced: [0.0008670079744768065, 0.08645073202108688, 0.91354249337886]
---------
Input: [6.7 3.3 5.7 2.5] Expected: [0. 0. 1.] Produced: [0.0007543327260544483, 0.05707602256712337, 0.9429665620506029]
---------
Input: [6.2 2.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.0031525859879632874, 0.8587658759553473, 0.140992039101056]
---------
Input: [4.9 3.  1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9961244820952206, 0.004192991344937267, 0.005721697452945688]
---------
Input: [5.8 2.8 5.1 2.4] Expected: [0. 0. 1.] Produced: [0.000754380691367376, 0.05711864120822936, 0.942920992791157]
---------
Input: [4.6 3.2 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9965040595700703, 0.003828637100922396, 0.006136838826481754]
---------
Input: [7.3 2.9 6.3 1.8] Expected: [0. 0. 1.] Produced: [0.0007543059967783323, 0.057108869056585077, 0.9429369323429295]
---------
Input: [5.1 3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9969358003573134, 0.003406802488863089, 0.006659656970735028]
---------
Input: [6.7 3.3 5.7 2.1] Expected: [0. 0. 1.] Produced: [0.0007544346647300632, 0.05709915252523929, 0.9429410692363566]
---------
Input: [4.7 3.2 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9971205734631425, 0.003227195175303928, 0.0070212226471389775]
---------
Input: [5.9 3.  5.1 1.8] Expected: [0. 0. 1.] Produced: [0.0007549430153713383, 0.057117396174778616, 0.9429054729566088]
---------
Input: [5.  3.3 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.997100314685636, 0.003246029386590045, 0.006951401848139891]
---------
Input: [5.5 2.4 3.8 1.1] Expected: [0. 1. 0.] Produced: [0.007015236528699541, 0.986684466666233, 0.012978853246214513]
---------
Input: [7.  3.2 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.006914549199686784, 0.9868620187079642, 0.012986660314991355]
---------
Input: [6.1 2.8 4.7 1.2] Expected: [0. 1. 0.] Produced: [0.006863286276414661, 0.9868305749603977, 0.013111501582546462]
---------
Input: [6.9 3.1 5.1 2.3] Expected: [0. 0. 1.] Produced: [0.0008204532925000001, 0.0733871563369888, 0.926624226960485]
---------
Input: [6.2 2.8 4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0009139919464791499, 0.10062153957008221, 0.8993413405663951]
---------
Input: [6.8 2.8 4.8 1.4] Expected: [0. 1. 0.] Produced: [0.006874058243846868, 0.9869222787738272, 0.013023446722892114]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9963082204353, 0.004019676759839833, 0.0059922746418539325]
---------
Input: [4.6 3.6 1.  0.2] Expected: [1. 0. 0.] Produced: [0.9981887606299782, 0.00214935180095203, 0.009950182669239018]
---------
Input: [5.8 2.7 5.1 1.9] Expected: [0. 0. 1.] Produced: [0.000754460671132505, 0.056975738278244654, 0.94306012603489]
---------
Input: [5.  3.5 1.6 0.6] Expected: [1. 0. 0.] Produced: [0.9931279760171445, 0.006903252316456657, 0.0035271870275262658]
---------
Input: [6.3 3.3 6.  2.5] Expected: [0. 0. 1.] Produced: [0.0007542893309874907, 0.05697020337508876, 0.9430750837697871]
---------
Input: [5.  2.  3.5 1. ] Expected: [0. 1. 0.] Produced: [0.007029683553910101, 0.9866095218595108, 0.013046111888836344]
---------
Input: [6.5 3.  5.2 2. ] Expected: [0. 0. 1.] Produced: [0.0007579205907618239, 0.0577116000534775, 0.9423147624781173]
---------
Input: [5.1 3.5 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9975154847703508, 0.0028335565624276237, 0.007772336299885156]
---------
Input: [6.3 2.5 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.0011399157537833572, 0.18568439595289837, 0.8142842031478277]
---------
Input: [5.7 2.5 5.  2. ] Expected: [0. 0. 1.] Produced: [0.0007544194794383558, 0.05753119105831797, 0.9425071340135938]
---------
Input: [5.  2.3 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.0074337283117851495, 0.9860106189593495, 0.012733532407388634]
---------
Input: [6.7 3.1 4.7 1.5] Expected: [0. 1. 0.] Produced: [0.0068909088862861055, 0.9869667918054518, 0.012938251705227378]
---------
Input: [6.3 2.7 4.9 1.8] Expected: [0. 0. 1.] Produced: [0.0008501039205515396, 0.08213864443824567, 0.9178673466840422]
---------
Input: [4.9 2.5 4.5 1.7] Expected: [0. 0. 1.] Produced: [0.000754953945319383, 0.05744760464691703, 0.9425589515475766]
---------
Input: [6.7 3.1 4.4 1.4] Expected: [0. 1. 0.] Produced: [0.00696446328518438, 0.9868299559635821, 0.012897280505585026]
---------
Input: [5.4 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.0027226794566994235, 0.7905489842697182, 0.20874057751800604]
---------
Input: [4.4 2.9 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9947927599291022, 0.005458630850498188, 0.004825740066016717]
---------
Input: [6.1 3.  4.9 1.8] Expected: [0. 0. 1.] Produced: [0.0010335134336803028, 0.1438495795103205, 0.856088708302614]
---------
Input: [6.9 3.2 5.7 2.3] Expected: [0. 0. 1.] Produced: [0.0007543783379014289, 0.0575263326084189, 0.9425165081873772]
---------
Input: [5.  3.4 1.6 0.4] Expected: [1. 0. 0.] Produced: [0.9947284194133353, 0.005488311894006252, 0.00441386934960245]
---------
Input: [6.  3.  4.8 1.8] Expected: [0. 0. 1.] Produced: [0.001131467561393328, 0.183208216411762, 0.816650736952121]
---------
Input: [5.  3.2 1.2 0.2] Expected: [1. 0. 0.] Produced: [0.9976211155307266, 0.0027296572327017367, 0.00805619992879358]
---------
Input: [5.  3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.996857498876177, 0.003485247201715425, 0.006543615144259679]
---------
Input: [6.4 2.7 5.3 1.9] Expected: [0. 0. 1.] Produced: [0.0007545344341576678, 0.0573882982782238, 0.9426511751226575]
---------
Input: [6.6 3.  4.4 1.4] Expected: [0. 1. 0.] Produced: [0.006940641116582393, 0.9868619285848915, 0.01292692849524138]
---------
Input: [5.6 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.006034447704073484, 0.980151604763205, 0.01973709929312284]
---------
Input: [6.  3.4 4.5 1.6] Expected: [0. 1. 0.] Produced: [0.006919886681836958, 0.9868937680190112, 0.012935341581081769]
---------
Input: [6.2 3.4 5.4 2.3] Expected: [0. 0. 1.] Produced: [0.0007544298909506735, 0.05734486928434642, 0.9426903546904765]
---------
Input: [6.3 3.4 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0007543386856666056, 0.057331778699122815, 0.9427099784235946]
---------
Input: [5.6 3.  4.1 1.3] Expected: [0. 1. 0.] Produced: [0.0069944880611460675, 0.9867625686172868, 0.012911815290541254]
---------
Input: [5.2 3.4 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9974069818454433, 0.0029432671098483577, 0.007523784893425705]
---------
Input: [5.6 2.9 3.6 1.3] Expected: [0. 1. 0.] Produced: [0.007478119119354471, 0.9858981963061411, 0.012554135174106808]
---------
Input: [6.4 2.8 5.6 2.1] Expected: [0. 0. 1.] Produced: [0.0007543061283425979, 0.05731796645729281, 0.942726933092668]
---------
Input: [7.7 3.8 6.7 2.2] Expected: [0. 0. 1.] Produced: [0.0007543712338466781, 0.057326481461134886, 0.942720107243697]
---------
Input: [6.7 3.1 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0007543170207144638, 0.0572840932475395, 0.9427592001998457]
---------
Input: [5.1 3.8 1.5 0.3] Expected: [1. 0. 0.] Produced: [0.9974194184812061, 0.0029290615547421478, 0.0074851754636589074]
---------
Input: [6.9 3.1 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.006876468274928575, 0.9869663163140604, 0.012971888943288745]
---------
Input: [5.1 3.8 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9973431737096179, 0.0030058445573800173, 0.007360853926533226]
---------
Input: [4.7 3.2 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9947574793313262, 0.005479029306429356, 0.004699239679509002]
---------
Input: [6.4 3.1 5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0007557278254328152, 0.05756023184172542, 0.9424772425051734]
---------
Input: [5.7 4.4 1.5 0.4] Expected: [1. 0. 0.] Produced: [0.9980701990094144, 0.0022711040912982024, 0.009386969722204382]
---------
Input: [5.4 3.4 1.7 0.2] Expected: [1. 0. 0.] Produced: [0.995924376697916, 0.004379184853047118, 0.005412250673327583]
---------
Input: [5.2 3.5 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9972214200936147, 0.0031268295799180795, 0.007136124758293965]
---------
Input: [4.8 3.  1.4 0.3] Expected: [1. 0. 0.] Produced: [0.9952734611118783, 0.004993258387567224, 0.0049336090010411735]
---------
Input: [5.2 2.7 3.9 1.4] Expected: [0. 1. 0.] Produced: [0.006924150285564689, 0.9865932391977935, 0.013159070824418518]
---------
Input: [5.1 3.3 1.7 0.5] Expected: [1. 0. 0.] Produced: [0.9904946258715313, 0.00922241179685833, 0.0029266131447637732]
---------
Input: [4.8 3.4 1.9 0.2] Expected: [1. 0. 0.] Produced: [0.9910333834548392, 0.008821010850330114, 0.0033989661501648994]
---------
Epoch 8000 RMSE = 0.12325972599413412
Epoch 8100 RMSE = 0.11651184255939395
Epoch 8200 RMSE = 0.12510052705193986
Epoch 8300 RMSE = 0.1368055766241553
Epoch 8400 RMSE = 0.13089634879447087
Epoch 8500 RMSE = 0.13414353618625932
Epoch 8600 RMSE = 0.12441034627342794
Epoch 8700 RMSE = 0.12398597275733098
Epoch 8800 RMSE = 0.123775071739837
Epoch 8900 RMSE = 0.11006189641728062
Input: [6.3 3.3 6.  2.5] Expected: [0. 0. 1.] Produced: [0.0007075961621678628, 0.0545205000290861, 0.9455124270463745]
---------
Input: [5.6 2.5 3.9 1.1] Expected: [0. 1. 0.] Produced: [0.006576035355477721, 0.9865006293258877, 0.013220735499453715]
---------
Input: [6.5 3.2 5.1 2. ] Expected: [0. 0. 1.] Produced: [0.0017250596556586996, 0.5073495691380266, 0.49261820799563033]
---------
Input: [5.1 3.5 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9976474933114652, 0.0026721355707921435, 0.00740162275503102]
---------
Input: [7.4 2.8 6.1 1.9] Expected: [0. 0. 1.] Produced: [0.0007076363829790886, 0.05399396056516681, 0.9460385221091628]
---------
Input: [5.1 3.8 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9974849475397585, 0.0028298740445392326, 0.007021889064699071]
---------
Input: [5.6 3.  4.1 1.3] Expected: [0. 1. 0.] Produced: [0.00656898741366788, 0.9864249723240363, 0.013279911256533342]
---------
Input: [4.3 3.  1.1 0.1] Expected: [1. 0. 0.] Produced: [0.9976647851153684, 0.0026588745785020742, 0.007581945125702583]
---------
Input: [5.  3.4 1.6 0.4] Expected: [1. 0. 0.] Produced: [0.9950260111894634, 0.005098438933415038, 0.004269190265135994]
---------
Input: [5.  2.  3.5 1. ] Expected: [0. 1. 0.] Produced: [0.006602658460097691, 0.9863441433587801, 0.013342935278664765]
---------
Input: [5.  3.5 1.6 0.6] Expected: [1. 0. 0.] Produced: [0.9935112886520066, 0.0064009281640297995, 0.003419068710052124]
---------
Input: [6.4 2.7 5.3 1.9] Expected: [0. 0. 1.] Produced: [0.000707737383986143, 0.05397535248278031, 0.9460510072539956]
---------
Input: [4.6 3.6 1.  0.2] Expected: [1. 0. 0.] Produced: [0.9982710980893272, 0.0020501124772239515, 0.009369649107458177]
---------
Input: [5.8 2.7 5.1 1.9] Expected: [0. 0. 1.] Produced: [0.0007077383631886441, 0.05395016177872654, 0.9460739010937703]
---------
Input: [7.7 3.8 6.7 2.2] Expected: [0. 0. 1.] Produced: [0.0007076177655759636, 0.053951762874193114, 0.9460808263207184]
---------
Input: [5.7 3.8 1.7 0.3] Expected: [1. 0. 0.] Produced: [0.9971619985445472, 0.003137640545818519, 0.0063467290099393035]
---------
Input: [4.7 3.2 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9950379764910067, 0.005109170862497381, 0.0045286156380603525]
---------
Input: [5.4 3.9 1.3 0.4] Expected: [1. 0. 0.] Produced: [0.9980415124675418, 0.0022806387833107598, 0.008440708130549517]
---------
Input: [4.4 2.9 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9950710114323347, 0.005088152938026443, 0.0046517953475343305]
---------
Input: [6.4 3.2 5.3 2.3] Expected: [0. 0. 1.] Produced: [0.0007077489564094058, 0.05392646958897022, 0.9460948468953323]
---------
Input: [6.6 2.9 4.6 1.3] Expected: [0. 1. 0.] Produced: [0.006482966657464223, 0.986584615186413, 0.013338392107881795]
---------
Input: [6.9 3.2 5.7 2.3] Expected: [0. 0. 1.] Produced: [0.000707644762335172, 0.053917281921319025, 0.9461118835079857]
---------
Input: [6.2 3.4 5.4 2.3] Expected: [0. 0. 1.] Produced: [0.0007077178168515928, 0.053896061497757164, 0.9461263692616407]
---------
Input: [4.8 3.1 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9945750163414937, 0.0055232670948275895, 0.004301271672583102]
---------
Input: [6.7 3.1 4.7 1.5] Expected: [0. 1. 0.] Produced: [0.006475987686714528, 0.9865939051367402, 0.013339179187640786]
---------
Input: [6.1 2.9 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.006406822207499514, 0.9862435965262545, 0.013727356743467773]
---------
Input: [4.8 3.  1.4 0.3] Expected: [1. 0. 0.] Produced: [0.9955425672714798, 0.004647173776193533, 0.004762050250193099]
---------
Input: [5.1 3.8 1.5 0.3] Expected: [1. 0. 0.] Produced: [0.9975585987673673, 0.0027569647238637114, 0.0071418961950385305]
---------
Input: [4.8 3.4 1.9 0.2] Expected: [1. 0. 0.] Produced: [0.9914453061740643, 0.008223500798425768, 0.0032824097628973755]
---------
Input: [4.5 2.3 1.3 0.3] Expected: [1. 0. 0.] Produced: [0.9888889415833164, 0.010409439291134793, 0.0030912275117433016]
---------
Input: [6.3 3.4 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0007076437316248111, 0.05388700991738786, 0.9461409721113223]
---------
Input: [6.1 3.  4.9 1.8] Expected: [0. 0. 1.] Produced: [0.0008006317400434678, 0.07802799737473513, 0.9219473827520607]
---------
Input: [5.1 3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9971069538447451, 0.003194260230438518, 0.00637820750374148]
---------
Input: [7.3 2.9 6.3 1.8] Expected: [0. 0. 1.] Produced: [0.0007075982067086191, 0.05384958310233133, 0.9461826991837805]
---------
Input: [4.6 3.2 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9966959947209589, 0.003586462960373178, 0.005883074858789427]
---------
Input: [4.9 2.4 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.0070349340445055, 0.9855139002831671, 0.013082335259892717]
---------
Input: [6.7 3.3 5.7 2.1] Expected: [0. 0. 1.] Produced: [0.0007077107442405504, 0.05384309103636264, 0.9461843770027714]
---------
Input: [4.9 3.  1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9963452784979452, 0.003914650105341164, 0.005503627944288068]
---------
Input: [6.4 2.8 5.6 2.1] Expected: [0. 0. 1.] Produced: [0.0007076172835664453, 0.05381945788733527, 0.9462111900426575]
---------
Input: [6.8 3.  5.5 2.1] Expected: [0. 0. 1.] Produced: [0.0007078792088445855, 0.05384961107362357, 0.9461766975074454]
---------
Input: [5.2 3.5 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9973751667893334, 0.0029360939282692135, 0.006824683832091985]
---------
Input: [5.  3.3 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.997260874373316, 0.003047233214615841, 0.006649226975364337]
---------
Input: [5.9 3.2 4.8 1.8] Expected: [0. 1. 0.] Produced: [0.0010736833430151846, 0.17899377622697502, 0.8208292247331194]
---------
Input: [6.3 2.5 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.0018367268943933767, 0.5571846782987535, 0.4428185691435244]
---------
Input: [6.3 2.7 4.9 1.8] Expected: [0. 0. 1.] Produced: [0.0011486217015008408, 0.217105557658732, 0.7828954559731244]
---------
Input: [5.9 3.  4.2 1.5] Expected: [0. 1. 0.] Produced: [0.0065259820624634516, 0.9866122615564358, 0.013191636801900993]
---------
Input: [6.  3.4 4.5 1.6] Expected: [0. 1. 0.] Produced: [0.006502370533361745, 0.9866568840434324, 0.013202480398446927]
---------
Input: [7.7 3.  6.1 2.3] Expected: [0. 0. 1.] Produced: [0.0007078830586468904, 0.05469443313891277, 0.9453374627898318]
---------
Input: [6.  2.2 5.  1.5] Expected: [0. 0. 1.] Produced: [0.0007083887594394565, 0.054741249666858526, 0.9452825416818669]
---------
Input: [6.4 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.006534134057029296, 0.9865940273590741, 0.013198636630783793]
---------
Input: [6.4 3.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.006504466879365364, 0.986650318804557, 0.013207237544781054]
---------
Input: [5.  3.  1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9943041629358736, 0.005775294170330701, 0.00416479758715241]
---------
Input: [6.2 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.006523444444693258, 0.9866148970238279, 0.013207458360354668]
---------
Input: [5.4 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.004608996728546058, 0.9610929962427394, 0.03879360661669937]
---------
Input: [6.5 3.  5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0007121471128825062, 0.05565005731296028, 0.944375614434151]
---------
Input: [7.2 3.2 6.  1.8] Expected: [0. 0. 1.] Produced: [0.0007627152254859835, 0.0685453228176645, 0.9314913429652725]
---------
Input: [6.7 2.5 5.8 1.8] Expected: [0. 0. 1.] Produced: [0.0007075971335047268, 0.05456430989305835, 0.945467543477715]
---------
Input: [4.6 3.1 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9954478429414895, 0.0047492484598305115, 0.004798010304751739]
---------
Input: [5.7 4.4 1.5 0.4] Expected: [1. 0. 0.] Produced: [0.9981649104630652, 0.002157537810810849, 0.008875093057892969]
---------
Input: [5.8 2.7 3.9 1.2] Expected: [0. 1. 0.] Produced: [0.006628345829231476, 0.9864086341824975, 0.013156946595996024]
---------
Input: [5.1 2.5 3.  1.1] Expected: [0. 1. 0.] Produced: [0.00851249140873312, 0.9827961359267497, 0.011865084538790004]
---------
Input: [5.5 3.5 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9980034953848091, 0.002320915329491112, 0.008359154870528367]
---------
Input: [7.  3.2 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.006496945983067242, 0.9866596476410862, 0.013219431580297912]
---------
Input: [5.4 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9977633280307007, 0.0025594247551776815, 0.007655904260898709]
---------
Input: [5.3 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9977248857643003, 0.0025974117760631317, 0.007561437949136852]
---------
Input: [5.  3.2 1.2 0.2] Expected: [1. 0. 0.] Produced: [0.9977473842398393, 0.0025763931829817397, 0.007664415160261359]
---------
Input: [5.  3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9970315137570087, 0.0032704000591452648, 0.00626224845671073]
---------
Input: [6.2 2.8 4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0020269292099358923, 0.6345024638065295, 0.3654048079308264]
---------
Input: [6.5 3.  5.2 2. ] Expected: [0. 0. 1.] Produced: [0.0007119167045500175, 0.05492596932208951, 0.9450894059914438]
---------
Input: [4.7 3.2 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.997277053240778, 0.0030333256819827705, 0.0067073480782132875]
---------
Input: [5.6 2.9 3.6 1.3] Expected: [0. 1. 0.] Produced: [0.007008279680530334, 0.985566901678934, 0.012918114302735645]
---------
Input: [5.6 2.7 4.2 1.3] Expected: [0. 1. 0.] Produced: [0.006487319389512719, 0.98649446960512, 0.013398781166495677]
---------
Input: [6.7 3.1 4.4 1.4] Expected: [0. 1. 0.] Produced: [0.006541896056229957, 0.9864735952230681, 0.0132836311222159]
---------
Input: [6.5 2.8 4.6 1.5] Expected: [0. 1. 0.] Produced: [0.006466039299149055, 0.9866125924030412, 0.013345115050346865]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.99651245538552, 0.003763121669848961, 0.005748158898015252]
---------
Input: [6.2 2.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.003780433791994848, 0.9284354620109874, 0.07151937859634573]
---------
Input: [5.8 2.8 5.1 2.4] Expected: [0. 0. 1.] Produced: [0.0007076406287570177, 0.05397396372609341, 0.9460529243733802]
---------
Input: [6.7 3.3 5.7 2.5] Expected: [0. 0. 1.] Produced: [0.0007075992445222516, 0.05396267879369068, 0.9460669736880071]
---------
Input: [6.1 2.8 4.7 1.2] Expected: [0. 1. 0.] Produced: [0.006458030707410455, 0.9865769585840132, 0.013393082472541689]
---------
Input: [5.6 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.005432485776217863, 0.9765952366652492, 0.023322324411481774]
---------
Input: [5.7 2.8 4.5 1.3] Expected: [0. 1. 0.] Produced: [0.006325668258444254, 0.9856051284653857, 0.014348568011328788]
---------
Input: [5.2 3.4 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9975484147172975, 0.0027688661627049273, 0.007181606125761532]
---------
Input: [6.9 3.1 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.0064629825996612485, 0.9866264928374304, 0.013339242155420261]
---------
Input: [5.8 2.7 4.1 1. ] Expected: [0. 1. 0.] Produced: [0.006573036448532769, 0.9864175342105239, 0.013313740804475897]
---------
Input: [5.2 2.7 3.9 1.4] Expected: [0. 1. 0.] Produced: [0.006499807182627138, 0.9862244974701769, 0.013557464486631067]
---------
Input: [6.  3.  4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0008739477146364513, 0.10108745112589153, 0.8988514811741075]
---------
Input: [5.9 3.  5.1 1.8] Expected: [0. 0. 1.] Produced: [0.0007081692401576791, 0.0539599629933658, 0.9460521355153054]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.996512347813126, 0.003762854750337018, 0.00574853325390167]
---------
Input: [7.9 3.8 6.4 2. ] Expected: [0. 0. 1.] Produced: [0.0017690168695823837, 0.5258839996371809, 0.4742536976045521]
---------
Input: [5.5 2.4 3.7 1. ] Expected: [0. 1. 0.] Produced: [0.00667403479435563, 0.9861269813222621, 0.013365794963976302]
---------
Input: [6.6 3.  4.4 1.4] Expected: [0. 1. 0.] Produced: [0.006520337816884931, 0.986420859677099, 0.013396157354362193]
---------
Input: [6.8 2.8 4.8 1.4] Expected: [0. 1. 0.] Produced: [0.006459546804272416, 0.9865270273696939, 0.013446169610392661]
---------
Input: [6.4 3.1 5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0007077378604698069, 0.05337200311743047, 0.9466519544455984]
---------
Input: [7.2 3.6 6.1 2.5] Expected: [0. 0. 1.] Produced: [0.0007075696502976969, 0.05335496170672195, 0.9466755607493942]
---------
Input: [5.  2.3 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.006967822387842498, 0.9855550946286342, 0.013208699609784323]
---------
Input: [4.9 2.5 4.5 1.7] Expected: [0. 0. 1.] Produced: [0.0007081295127869913, 0.05330534349359894, 0.9466920616890957]
---------
Input: [5.4 3.4 1.7 0.2] Expected: [1. 0. 0.] Produced: [0.9961594614145128, 0.0040756768953507175, 0.005223257168566739]
---------
Input: [5.5 2.4 3.8 1.1] Expected: [0. 1. 0.] Produced: [0.0065871073967999835, 0.986286661701417, 0.013405401962150074]
---------
Input: [7.7 2.6 6.9 2.3] Expected: [0. 0. 1.] Produced: [0.0007075387392258349, 0.05333054031418717, 0.9467020960409819]
---------
Input: [6.3 2.8 5.1 1.5] Expected: [0. 0. 1.] Produced: [0.0007585373290135213, 0.06575009890361884, 0.9342567515523321]
---------
Input: [5.1 3.3 1.7 0.5] Expected: [1. 0. 0.] Produced: [0.9909897549277823, 0.008512232279585951, 0.0028523759206236367]
---------
Input: [5.7 2.5 5.  2. ] Expected: [0. 0. 1.] Produced: [0.0007076651032539885, 0.05328868438502782, 0.9467366784367247]
---------
Input: [6.9 3.1 5.1 2.3] Expected: [0. 0. 1.] Produced: [0.0007162847808691841, 0.05522487796185532, 0.9447790390832823]
---------
Input: [5.  3.6 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9977374814310582, 0.0025816617566187397, 0.007623264600144382]
---------
Input: [6.7 3.1 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.000707589578388264, 0.05326548341930823, 0.9467634502561003]
---------
Epoch 9000 RMSE = 0.11198931842526896
Epoch 9100 RMSE = 0.12506459216701746
Epoch 9200 RMSE = 0.1343236774589586
Epoch 9300 RMSE = 0.12110089329915517
Epoch 9400 RMSE = 0.11997534732298341
Epoch 9500 RMSE = 0.144588125803995
Epoch 9600 RMSE = 0.11034094709001437
Epoch 9700 RMSE = 0.12942941581725756
Epoch 9800 RMSE = 0.12501520632841037
Epoch 9900 RMSE = 0.12493071278839948
Input: [6.2 2.8 4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0007081626379999291, 0.0699740496640724, 0.9299795234660644]
---------
Input: [5.1 3.8 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9976035659344712, 0.002707769935303419, 0.006661445517156252]
---------
Input: [5.8 2.7 3.9 1.2] Expected: [0. 1. 0.] Produced: [0.006265194153081729, 0.9869420103855945, 0.012677498724714837]
---------
Input: [5.8 2.7 4.1 1. ] Expected: [0. 1. 0.] Produced: [0.0062148717516690094, 0.9870419800050858, 0.012733363113951988]
---------
Input: [6.8 2.8 4.8 1.4] Expected: [0. 1. 0.] Produced: [0.006113216184424911, 0.987233147275219, 0.012765058411538117]
---------
Input: [5.8 2.7 5.1 1.9] Expected: [0. 0. 1.] Produced: [0.0006685377555886786, 0.058986993193742863, 0.9410139616344964]
---------
Input: [6.5 3.  5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0006685815936754866, 0.0589883192521261, 0.9410143002583066]
---------
Input: [5.6 2.9 3.6 1.3] Expected: [0. 1. 0.] Produced: [0.0066124024635394124, 0.9862541498031061, 0.012365655497438113]
---------
Input: [5.6 2.7 4.2 1.3] Expected: [0. 1. 0.] Produced: [0.006116144656505815, 0.9869676563195997, 0.012956279360347295]
---------
Input: [5.  2.  3.5 1. ] Expected: [0. 1. 0.] Produced: [0.006238406674688797, 0.9869462014527653, 0.012786915126875971]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.9966811112946885, 0.0035966763052124383, 0.005461410613306005]
---------
Input: [4.6 3.6 1.  0.2] Expected: [1. 0. 0.] Produced: [0.9983409419789261, 0.001973648747901837, 0.008834547197539159]
---------
Input: [4.6 3.1 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9956625216878697, 0.004536634963356913, 0.004563215519729933]
---------
Input: [5.7 4.4 1.5 0.4] Expected: [1. 0. 0.] Produced: [0.9982440082606385, 0.0020711979023180925, 0.008392537259902709]
---------
Input: [4.8 3.1 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9948297610254868, 0.00528610948988282, 0.004084628136922947]
---------
Input: [7.2 3.2 6.  1.8] Expected: [0. 0. 1.] Produced: [0.0006690557005983849, 0.059120902346559166, 0.9408864029434806]
---------
Input: [5.9 3.2 4.8 1.8] Expected: [0. 1. 0.] Produced: [0.0007467320482642838, 0.0816230515067387, 0.9182686825415124]
---------
Input: [5.1 3.5 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9977580973019186, 0.0025579048414557954, 0.007018222459267674]
---------
Input: [5.6 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.0036400094114241747, 0.935409611853361, 0.06445884695232143]
---------
Input: [6.3 3.4 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0006684533586047766, 0.0593267481987296, 0.9406783407675228]
---------
Input: [6.4 3.1 5.5 1.8] Expected: [0. 0. 1.] Produced: [0.0006685950221190861, 0.05932216863169076, 0.9406790142392648]
---------
Input: [5.1 3.3 1.7 0.5] Expected: [1. 0. 0.] Produced: [0.9913918743346386, 0.00819267807944719, 0.0026959256917208443]
---------
Input: [7.9 3.8 6.4 2. ] Expected: [0. 0. 1.] Produced: [0.0008298043485542531, 0.1121212223101244, 0.8879114427637241]
---------
Input: [5.4 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9978676307540709, 0.0024490508967627018, 0.007262699915319983]
---------
Input: [4.5 2.3 1.3 0.3] Expected: [1. 0. 0.] Produced: [0.9893548560905343, 0.010016283798376747, 0.002927614292641637]
---------
Input: [6.1 2.9 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.005947591803919983, 0.9860868227410071, 0.013909117100757488]
---------
Input: [5.5 3.5 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9980938168172627, 0.002223876757294602, 0.007918701279471514]
---------
Input: [5.1 2.5 3.  1.1] Expected: [0. 1. 0.] Produced: [0.007997835477303717, 0.9836021524410681, 0.011418160081161758]
---------
Input: [5.  3.  1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9945780243144787, 0.005513484052261485, 0.003965487254083861]
---------
Input: [6.8 3.  5.5 2.1] Expected: [0. 0. 1.] Produced: [0.000668511087050204, 0.05923797644328448, 0.9407655134213556]
---------
Input: [5.2 3.5 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9975017264027024, 0.002808044937587357, 0.0064776264470048995]
---------
Input: [4.9 2.4 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.006637147359977355, 0.9862517687958485, 0.012472818072788628]
---------
Input: [5.  3.2 1.2 0.2] Expected: [1. 0. 0.] Produced: [0.9978529764638349, 0.0024646980067733557, 0.007272041543139997]
---------
Input: [7.7 3.  6.1 2.3] Expected: [0. 0. 1.] Produced: [0.0006684124687779821, 0.059217812972880646, 0.9407913203668059]
---------
Input: [5.2 3.4 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9976660378503837, 0.0026480846909349913, 0.006816282577611787]
---------
Input: [6.4 2.8 5.6 2.1] Expected: [0. 0. 1.] Produced: [0.0006684293014901377, 0.0591971773354133, 0.9408107828465833]
---------
Input: [6.5 3.  5.2 2. ] Expected: [0. 0. 1.] Produced: [0.0006692868208807046, 0.059347008969945984, 0.940645074324527]
---------
Input: [6.6 2.9 4.6 1.3] Expected: [0. 1. 0.] Produced: [0.006133105313587046, 0.9872197849532544, 0.01273242522563753]
---------
Input: [6.4 3.2 5.3 2.3] Expected: [0. 0. 1.] Produced: [0.0006685267623685897, 0.059154071208746765, 0.9408440265932088]
---------
Input: [5.4 3.4 1.7 0.2] Expected: [1. 0. 0.] Produced: [0.9963531874865623, 0.0038949373892767035, 0.004963618336517749]
---------
Input: [6.7 3.1 5.6 2.4] Expected: [0. 0. 1.] Produced: [0.0006684358344649254, 0.05914186354578406, 0.9408647105760928]
---------
Input: [7.  3.2 4.7 1.4] Expected: [0. 1. 0.] Produced: [0.006145840910261551, 0.9871904426288772, 0.01272085688140564]
---------
Input: [5.4 3.9 1.3 0.4] Expected: [1. 0. 0.] Produced: [0.9981298280849221, 0.0021866266001647912, 0.00799090082950526]
---------
Input: [4.7 3.2 1.3 0.2] Expected: [1. 0. 0.] Produced: [0.9974074342037321, 0.002901092941564953, 0.006366059589782346]
---------
Input: [4.8 3.4 1.9 0.2] Expected: [1. 0. 0.] Produced: [0.9917887665518109, 0.007922781386159988, 0.003102219355457881]
---------
Input: [5.3 3.7 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9978314945943928, 0.0024846310910313597, 0.0071750879257545435]
---------
Input: [5.1 3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.997248093779218, 0.003052981657178583, 0.006057931224495507]
---------
Input: [6.7 3.3 5.7 2.5] Expected: [0. 0. 1.] Produced: [0.000668430156562, 0.05912431955016403, 0.9408824060391772]
---------
Input: [6.6 3.  4.4 1.4] Expected: [0. 1. 0.] Produced: [0.006167161807732901, 0.9871469020034115, 0.012709240788124986]
---------
Input: [5.5 2.4 3.8 1.1] Expected: [0. 1. 0.] Produced: [0.006227980868288992, 0.987030655145926, 0.012710508359133125]
---------
Input: [4.3 3.  1.1 0.1] Expected: [1. 0. 0.] Produced: [0.9977713656726586, 0.002547891840458989, 0.0071807531265142095]
---------
Input: [6.3 3.3 6.  2.5] Expected: [0. 0. 1.] Produced: [0.0006684008623837101, 0.05910901155037826, 0.9409002668883848]
---------
Input: [6.9 3.2 5.7 2.3] Expected: [0. 0. 1.] Produced: [0.0006684399308988595, 0.05908817011373823, 0.9409178302434646]
---------
Input: [6.3 2.7 4.9 1.8] Expected: [0. 0. 1.] Produced: [0.000677540812871942, 0.06140787259302082, 0.9385697735421008]
---------
Input: [5.9 3.  5.1 1.8] Expected: [0. 0. 1.] Produced: [0.0006687152779143087, 0.05904491190408315, 0.9409437378809375]
---------
Input: [5.1 3.8 1.5 0.3] Expected: [1. 0. 0.] Produced: [0.9976750032102156, 0.002637175298722133, 0.006777041651118584]
---------
Input: [5.  3.5 1.6 0.6] Expected: [1. 0. 0.] Produced: [0.9938345657800463, 0.00611022371967232, 0.0032541020550857155]
---------
Input: [6.7 3.1 4.7 1.5] Expected: [0. 1. 0.] Produced: [0.006126795639593635, 0.987216956706224, 0.012744478808507959]
---------
Input: [4.7 3.2 1.6 0.2] Expected: [1. 0. 0.] Produced: [0.9952730856734032, 0.004888356765119002, 0.0043004244885344045]
---------
Input: [5.6 2.5 3.9 1.1] Expected: [0. 1. 0.] Produced: [0.006217322804978411, 0.9870431148185652, 0.01272120175263223]
---------
Input: [6.4 2.7 5.3 1.9] Expected: [0. 0. 1.] Produced: [0.0006684972563620855, 0.059029918128340284, 0.940972975880129]
---------
Input: [6.7 3.3 5.7 2.1] Expected: [0. 0. 1.] Produced: [0.0006684652928341305, 0.05901470631306859, 0.9409897678566763]
---------
Input: [5.  3.4 1.5 0.2] Expected: [1. 0. 0.] Produced: [0.9971769791129617, 0.0031208702194665313, 0.005957248939886265]
---------
Input: [4.8 3.  1.4 0.3] Expected: [1. 0. 0.] Produced: [0.9957667542413526, 0.004434724470504933, 0.004531923652172215]
---------
Input: [7.7 2.6 6.9 2.3] Expected: [0. 0. 1.] Produced: [0.0006683878300970111, 0.058999910944420184, 0.9410100722133925]
---------
Input: [6.  3.4 4.5 1.6] Expected: [0. 1. 0.] Produced: [0.006149131448592777, 0.9871541559245186, 0.012738716149031354]
---------
Input: [6.7 2.5 5.8 1.8] Expected: [0. 0. 1.] Produced: [0.0006684102964820976, 0.05898042632983244, 0.9410284925108793]
---------
Input: [6.9 3.1 5.1 2.3] Expected: [0. 0. 1.] Produced: [0.0006860447251050296, 0.0636828836141301, 0.9362974223801747]
---------
Input: [6.4 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.006179376918646061, 0.9871047232472585, 0.012726125358887388]
---------
Input: [6.2 2.9 4.3 1.3] Expected: [0. 1. 0.] Produced: [0.006169668621275405, 0.9871238436921412, 0.012734532328531058]
---------
Input: [7.2 3.6 6.1 2.5] Expected: [0. 0. 1.] Produced: [0.0006684077453712795, 0.058942208280793267, 0.941065703798405]
---------
Input: [6.9 3.1 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.006115215070013918, 0.9872265504908001, 0.012764656989363058]
---------
Input: [5.6 3.  4.1 1.3] Expected: [0. 1. 0.] Produced: [0.006210728307185988, 0.9870404682340118, 0.012710549043851169]
---------
Input: [5.  3.3 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9973937993033631, 0.0029126299114982726, 0.0063145392864654825]
---------
Input: [5.9 3.  4.2 1.5] Expected: [0. 1. 0.] Produced: [0.0061714474351252996, 0.9871136039005822, 0.012727627603237202]
---------
Input: [5.  2.3 3.3 1. ] Expected: [0. 1. 0.] Produced: [0.006576133449510609, 0.9863359033022897, 0.012547677759876075]
---------
Input: [4.6 3.2 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9968547579242046, 0.003429334792832595, 0.005585803117739238]
---------
Input: [6.5 3.2 5.1 2. ] Expected: [0. 0. 1.] Produced: [0.0007058647311012309, 0.06927555547415815, 0.9306937579434821]
---------
Input: [5.  3.6 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9978418809608207, 0.002474374560639369, 0.007219678044158706]
---------
Input: [6.3 2.8 5.1 1.5] Expected: [0. 0. 1.] Produced: [0.0007529363738304746, 0.08388852382640843, 0.9160956473204372]
---------
Input: [7.4 2.8 6.1 1.9] Expected: [0. 0. 1.] Produced: [0.0006684009184574128, 0.05886802150750919, 0.9411410799255066]
---------
Input: [4.4 2.9 1.4 0.2] Expected: [1. 0. 0.] Produced: [0.995304148963157, 0.004866754301407728, 0.0044184803578592394]
---------
Input: [5.7 3.8 1.7 0.3] Expected: [1. 0. 0.] Produced: [0.9973063784492439, 0.002991930242069442, 0.006041563306514723]
---------
Input: [5.8 2.8 5.1 2.4] Expected: [0. 0. 1.] Produced: [0.0006684526205870605, 0.058844735924439874, 0.9411591032641244]
---------
Input: [5.2 2.7 3.9 1.4] Expected: [0. 1. 0.] Produced: [0.006083016552472367, 0.986394547024711, 0.01342164384430385]
---------
Input: [6.1 3.  4.9 1.8] Expected: [0. 0. 1.] Produced: [0.0006884840643327592, 0.064116582986243, 0.9358402236294717]
---------
Input: [6.4 3.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.00615241083707719, 0.9871415415507132, 0.012749558703882457]
---------
Input: [4.9 3.1 1.5 0.1] Expected: [1. 0. 0.] Produced: [0.996682521634786, 0.0035946535502836587, 0.005463919558704831]
---------
Input: [4.9 2.5 4.5 1.7] Expected: [0. 0. 1.] Produced: [0.0006688907368182779, 0.058773259726384745, 0.9412002671408256]
---------
Input: [6.  3.  4.8 1.8] Expected: [0. 0. 1.] Produced: [0.0007014887533448421, 0.06771074509391321, 0.9322252721104242]
---------
Input: [5.4 3.  4.5 1.5] Expected: [0. 1. 0.] Produced: [0.0010370429766695252, 0.20191986381659052, 0.7976225186129516]
---------
Input: [4.9 3.  1.4 0.2] Expected: [1. 0. 0.] Produced: [0.9965256484589791, 0.0037421165976152583, 0.005227543177808869]
---------
Input: [5.7 2.8 4.5 1.3] Expected: [0. 1. 0.] Produced: [0.005917440401665729, 0.9858256508006622, 0.014155013039365996]
---------
Input: [5.5 2.4 3.7 1. ] Expected: [0. 1. 0.] Produced: [0.006306796694026199, 0.9869233984557467, 0.012635964272275074]
---------
Input: [7.3 2.9 6.3 1.8] Expected: [0. 0. 1.] Produced: [0.000668388708596438, 0.05941571324443185, 0.9405931752035649]
---------
Input: [5.  3.4 1.6 0.4] Expected: [1. 0. 0.] Produced: [0.9952738096251093, 0.004871371433557898, 0.004058327709126966]
---------
Input: [6.2 2.2 4.5 1.5] Expected: [0. 1. 0.] Produced: [0.0028726177420118995, 0.8723006230130249, 0.12774092059414416]
---------
Input: [6.3 2.5 4.9 1.5] Expected: [0. 1. 0.] Produced: [0.000969920997772653, 0.17257344355489834, 0.827421858670639]
---------
Input: [5.7 2.5 5.  2. ] Expected: [0. 0. 1.] Produced: [0.0006684815868440728, 0.060050898624137476, 0.9399511113708775]
---------
Input: [6.2 3.4 5.4 2.3] Expected: [0. 0. 1.] Produced: [0.000668495406870014, 0.06003150423768337, 0.9399669004939352]
---------
Input: [6.5 2.8 4.6 1.5] Expected: [0. 1. 0.] Produced: [0.006118379875408283, 0.9873581555589324, 0.012625779924175094]
---------
Input: [6.1 2.8 4.7 1.2] Expected: [0. 1. 0.] Produced: [0.006113343595998202, 0.9873416883206497, 0.012653430083614092]
---------
Input: [7.7 3.8 6.7 2.2] Expected: [0. 0. 1.] Produced: [0.0006684134555493419, 0.060029758984844625, 0.9399790106815685]
---------
Input: [6.7 3.1 4.4 1.4] Expected: [0. 1. 0.] Produced: [0.00618634022235496, 0.9872276829144144, 0.01257420755842553]
---------
Input: [6.  2.2 5.  1.5] Expected: [0. 0. 1.] Produced: [0.0006686406706671902, 0.060013744414893905, 0.9399863074117155]
---------
Epoch 10000 RMSE = 0.12178022773183571
Final RMSE = 0.12178022773183571
---TESTING---
Input values: [7.2 3.  5.8 1.6]
Expected output: [0. 0. 1.]
Actual output: 0.0012217269728460442 0.3068335294189689 0.6933047724972411 
---------
Input values: [5.5 2.3 4.  1.3]
Expected output: [0. 1. 0.]
Actual output: 0.006053143479261879 0.9866766424992632 0.013262051487098227 
---------
Input values: [4.4 3.  1.3 0.2]
Expected output: [1. 0. 0.]
Actual output: 0.996663257147747 0.0036168476685979685 0.005414021100157961 
---------
Input values: [4.8 3.  1.4 0.1]
Expected output: [1. 0. 0.]
Actual output: 0.9968944158149886 0.003399787838875873 0.0057181904101604045 
---------
Input values: [4.9 3.1 1.5 0.1]
Expected output: [1. 0. 0.]
Actual output: 0.9966808709188507 0.0036012639675383736 0.005453957569882452 
---------
Input values: [4.4 3.2 1.3 0.2]
Expected output: [1. 0. 0.]
Actual output: 0.997207307168425 0.003097274621989796 0.006062763172643193 
---------
Input values: [5.5 4.2 1.4 0.2]
Expected output: [1. 0. 0.]
Actual output: 0.9983662474424536 0.0019479616860958251 0.008915421738251609 
---------
Input values: [5.7 2.9 4.2 1.3]
Expected output: [0. 1. 0.]
Actual output: 0.006170014080504732 0.9872150063040352 0.01264575404577938 
---------
Input values: [6.1 2.6 5.6 1.4]
Expected output: [0. 0. 1.]
Actual output: 0.0006684425716991519 0.05966292606793034 0.9403433203550509 
---------
Input values: [6.3 2.5 5.  1.9]
Expected output: [0. 0. 1.]
Actual output: 0.0006686541382447344 0.05966474486663886 0.940331188304395 
---------
Input values: [5.4 3.9 1.7 0.4]
Expected output: [1. 0. 0.]
Actual output: 0.996953428875973 0.0033292464347394893 0.005468319384462989 
---------
Input values: [5.2 4.1 1.5 0.1]
Expected output: [1. 0. 0.]
Actual output: 0.9982562199089646 0.0020605012213743886 0.00848580279501579 
---------
Input values: [6.7 3.  5.2 2.3]
Expected output: [0. 0. 1.]
Actual output: 0.0006687969897078033 0.05969273840511174 0.9403018223275732 
---------
Input values: [5.1 3.8 1.9 0.4]
Expected output: [1. 0. 0.]
Actual output: 0.9940349716318357 0.005959322493792661 0.0034120199598077095 
---------
Input values: [5.7 2.6 3.5 1. ]
Expected output: [0. 1. 0.]
Actual output: 0.006773715833132904 0.9860430326952131 0.012266372309407591 
---------
Input values: [4.8 3.4 1.6 0.2]
Expected output: [1. 0. 0.]
Actual output: 0.9963862786249756 0.0038708836187065655 0.00504201717067704 
---------
Input values: [6.  2.2 4.  1. ]
Expected output: [0. 1. 0.]
Actual output: 0.006174655422869829 0.9872042377587796 0.012674079599650133 
---------
Input values: [5.1 3.7 1.5 0.4]
Expected output: [1. 0. 0.]
Actual output: 0.9972235783182604 0.0030740647225202798 0.005893906132714911 
---------
Input values: [6.3 2.9 5.6 1.8]
Expected output: [0. 0. 1.]
Actual output: 0.0006684305145401107 0.05960801447247951 0.9403973593819956 
---------
Input values: [4.6 3.4 1.4 0.3]
Expected output: [1. 0. 0.]
Actual output: 0.9969849990732786 0.003305364717091499 0.005650331877932 
---------
Input values: [5.  3.5 1.3 0.3]
Expected output: [1. 0. 0.]
Actual output: 0.9977786525042766 0.0025376840274367557 0.007032512949098013 
---------
Input values: [5.6 2.8 4.9 2. ]
Expected output: [0. 0. 1.]
Actual output: 0.0006685943335800649 0.05957629767249776 0.9404143813125526 
---------
Input values: [6.7 3.  5.  1.7]
Expected output: [0. 1. 0.]
Actual output: 0.005925884593735719 0.9860685005309234 0.013947097127103528 
---------
Input values: [6.  2.7 5.1 1.6]
Expected output: [0. 1. 0.]
Actual output: 0.0006690310245121983 0.05966955396971063 0.9403234865735878 
---------
Input values: [5.5 2.5 4.  1.3]
Expected output: [0. 1. 0.]
Actual output: 0.00614886759452265 0.9872009916126026 0.012695768800788867 
---------
Input values: [7.7 2.8 6.7 2. ]
Expected output: [0. 0. 1.]
Actual output: 0.0006683651052877807 0.05987218480422054 0.9401368944952316 
---------
Input values: [7.6 3.  6.6 2.1]
Expected output: [0. 0. 1.]
Actual output: 0.0006683665231615917 0.059853110038213954 0.94015586230502 
---------
Input values: [5.5 2.6 4.4 1.2]
Expected output: [0. 1. 0.]
Actual output: 0.005463498514792244 0.9818072951160974 0.018171008784130353 
---------
Input values: [6.1 3.  4.6 1.4]
Expected output: [0. 1. 0.]
Actual output: 0.0061213236707612275 0.9873039063218793 0.01266781870589789 
---------
Input values: [5.7 3.  4.2 1.2]
Expected output: [0. 1. 0.]
Actual output: 0.0061978435391604495 0.9871829281909903 0.012612040833399497 
---------
Input values: [5.4 3.4 1.5 0.4]
Expected output: [1. 0. 0.]
Actual output: 0.9966978345669004 0.003572203617249037 0.0051887210303639 
---------
Input values: [6.8 3.2 5.9 2.3]
Expected output: [0. 0. 1.]
Actual output: 0.0006683831763913509 0.05983446266512096 0.9401728621809385 
---------
Input values: [5.8 4.  1.2 0.2]
Expected output: [1. 0. 0.]
Actual output: 0.9984631445512299 0.0018484253948386614 0.00936517389631214 
---------
Input values: [5.1 3.5 1.4 0.3]
Expected output: [1. 0. 0.]
Actual output: 0.9975290582700074 0.002782334120032027 0.006486630858858119 
---------
Input values: [6.1 2.8 4.  1.3]
Expected output: [0. 1. 0.]
Actual output: 0.006248904311169117 0.9870810706693222 0.012565509397166883 
---------
Input values: [6.5 3.  5.8 2.2]
Expected output: [0. 0. 1.]
Actual output: 0.0006683811399580266 0.05981604770472638 0.9401915247112383 
---------
Input values: [6.  2.9 4.5 1.5]
Expected output: [0. 1. 0.]
Actual output: 0.00609529697446185 0.9871400133673232 0.012833633321859177 
---------
Input values: [7.1 3.  5.9 2.1]
Expected output: [0. 0. 1.]
Actual output: 0.0006683895060700597 0.05979841568326377 0.9402087787044314 
---------
Input values: [6.3 3.3 4.7 1.6]
Expected output: [0. 1. 0.]
Actual output: 0.0061243440982124645 0.987308520382086 0.012653891026733043 
---------
Input values: [5.7 2.8 4.1 1.3]
Expected output: [0. 1. 0.]
Actual output: 0.0061815544436220025 0.9872069796306323 0.012625010413443449 
---------
Input values: [6.3 2.3 4.4 1.3]
Expected output: [0. 1. 0.]
Actual output: 0.00611394904267241 0.9873057914662262 0.012686137059111029 
---------
Input values: [5.8 2.6 4.  1.2]
Expected output: [0. 1. 0.]
Actual output: 0.006198163175609239 0.9871768288844275 0.01262251395794774 
---------
Input values: [6.4 2.8 5.6 2.2]
Expected output: [0. 0. 1.]
Actual output: 0.0006683810813299653 0.05978001530647202 0.9402271327813433 
---------
Input values: [6.9 3.1 5.4 2.1]
Expected output: [0. 0. 1.]
Actual output: 0.0006757220411522212 0.06172143465049009 0.9382752913410906 
---------
Input values: [5.8 2.7 5.1 1.9]
Expected output: [0. 0. 1.]
Actual output: 0.0006684883684393913 0.05973296430902678 0.9402666345644802 
---------

RMSE = 0.0072707169385630615

"""

"""
---run_sin() sample run---
---TRAINING---
Input: [1.34] Expected: [0.9734845] Produced: [0.7502242620209626]
---------
Input: [1.5] Expected: [0.997495] Produced: [0.754567150095591]
---------
Input: [0.45] Expected: [0.43496552] Produced: [0.7241804935375604]
---------
Input: [0.1] Expected: [0.09983341] Produced: [0.7107505398755638]
---------
Input: [0.02] Expected: [0.01999867] Produced: [0.7067026100720601]
---------
Input: [0.18] Expected: [0.17902957] Produced: [0.7116019332436516]
---------
Input: [1.53] Expected: [0.9991679] Produced: [0.7517320118383726]
---------
Input: [0.01] Expected: [0.00999983] Produced: [0.7047856637264955]
---------
Input: [0.24] Expected: [0.23770262] Produced: [0.7122314895077773]
---------
Input: [1.47] Expected: [0.99492437] Produced: [0.7487286368461821]
---------
Input: [0.64] Expected: [0.59719545] Produced: [0.7259342490226127]
---------
Input: [1.49] Expected: [0.9967378] Produced: [0.7495178836151102]
---------
Input: [0.07] Expected: [0.06994285] Produced: [0.7058223791463668]
---------
Input: [0.65] Expected: [0.6051864] Produced: [0.7254117424377832]
---------
Input: [0.36] Expected: [0.35227424] Produced: [0.7151825458623682]
---------
Input: [1.38] Expected: [0.98185354] Produced: [0.7453221471798624]
---------
Epoch 0 RMSE = 0.4209215643946678
Epoch 100 RMSE = 0.37429942591679716
Epoch 200 RMSE = 0.3400689389175714
Epoch 300 RMSE = 0.27834224857769535
Epoch 400 RMSE = 0.22627398691789324
Epoch 500 RMSE = 0.19014981187948823
Epoch 600 RMSE = 0.16465851648231142
Epoch 700 RMSE = 0.14589414527498398
Epoch 800 RMSE = 0.1315232959070727
Epoch 900 RMSE = 0.12016172928987086
Input: [0.18] Expected: [0.17902957] Produced: [0.2628854600111096]
---------
Input: [0.01] Expected: [0.00999983] Produced: [0.15346644042431773]
---------
Input: [0.1] Expected: [0.09983341] Produced: [0.20627156289975979]
---------
Input: [0.65] Expected: [0.6051864] Produced: [0.629335540026271]
---------
Input: [0.64] Expected: [0.59719545] Produced: [0.6230469021949155]
---------
Input: [0.24] Expected: [0.23770262] Produced: [0.309252341410936]
---------
Input: [0.45] Expected: [0.43496552] Produced: [0.48460450535090144]
---------
Input: [1.34] Expected: [0.9734845] Produced: [0.8420617237707313]
---------
Input: [1.38] Expected: [0.98185354] Produced: [0.8467931665229972]
---------
Input: [0.36] Expected: [0.35227424] Produced: [0.4098471113398985]
---------
Input: [1.5] Expected: [0.997495] Produced: [0.8581574163083306]
---------
Input: [0.07] Expected: [0.06994285] Produced: [0.18727713438673793]
---------
Input: [1.53] Expected: [0.9991679] Produced: [0.8605867461204141]
---------
Input: [1.49] Expected: [0.9967378] Produced: [0.8575739740810918]
---------
Input: [0.02] Expected: [0.01999867] Produced: [0.1587312130776954]
---------
Input: [1.47] Expected: [0.99492437] Produced: [0.8559406304391305]
---------
Epoch 1000 RMSE = 0.11094005114231272
Epoch 1100 RMSE = 0.10330277115436305
Epoch 1200 RMSE = 0.09687151532688074
Epoch 1300 RMSE = 0.09138022373111018
Epoch 1400 RMSE = 0.08663908093993802
Epoch 1500 RMSE = 0.08250686399021413
Epoch 1600 RMSE = 0.07887678315402206
Epoch 1700 RMSE = 0.07566467822460843
Epoch 1800 RMSE = 0.07280667097754742
Epoch 1900 RMSE = 0.07024863479559536
Input: [1.53] Expected: [0.9991679] Produced: [0.907380493505203]
---------
Input: [0.01] Expected: [0.00999983] Produced: [0.09828701406483618]
---------
Input: [1.34] Expected: [0.9734845] Produced: [0.889841347555426]
---------
Input: [0.24] Expected: [0.23770262] Produced: [0.2527595880767969]
---------
Input: [0.36] Expected: [0.35227424] Produced: [0.3695804619298087]
---------
Input: [0.07] Expected: [0.06994285] Produced: [0.12833443516601778]
---------
Input: [1.47] Expected: [0.99492437] Produced: [0.9027501666648982]
---------
Input: [1.5] Expected: [0.997495] Produced: [0.9052311636868745]
---------
Input: [1.38] Expected: [0.98185354] Produced: [0.8944277390286602]
---------
Input: [0.02] Expected: [0.01999867] Produced: [0.10289484172415657]
---------
Input: [0.64] Expected: [0.59719545] Produced: [0.6376833876913365]
---------
Input: [0.1] Expected: [0.09983341] Produced: [0.146052260914984]
---------
Input: [0.18] Expected: [0.17902957] Produced: [0.20223835440566798]
---------
Input: [0.65] Expected: [0.6051864] Produced: [0.6451355663151299]
---------
Input: [0.45] Expected: [0.43496552] Produced: [0.4622968894642116]
---------
Input: [1.49] Expected: [0.9967378] Produced: [0.9043144455124777]
---------
Epoch 2000 RMSE = 0.06794961026858465
Epoch 2100 RMSE = 0.06587390222053446
Epoch 2200 RMSE = 0.0639923766950728
Epoch 2300 RMSE = 0.06228141237159503
Epoch 2400 RMSE = 0.060720174670142575
Epoch 2500 RMSE = 0.05929173644488139
Epoch 2600 RMSE = 0.05798090735484517
Epoch 2700 RMSE = 0.05677503042343324
Epoch 2800 RMSE = 0.05566265167459291
Epoch 2900 RMSE = 0.05463524984706746
Input: [0.36] Expected: [0.35227424] Produced: [0.3505097005061572]
---------
Input: [0.24] Expected: [0.23770262] Produced: [0.23022151144323713]
---------
Input: [0.1] Expected: [0.09983341] Produced: [0.12620416419447553]
---------
Input: [0.64] Expected: [0.59719545] Produced: [0.6402975277371117]
---------
Input: [1.53] Expected: [0.9991679] Produced: [0.9263206279421109]
---------
Input: [0.01] Expected: [0.00999983] Produced: [0.08203656922025637]
---------
Input: [1.5] Expected: [0.997495] Produced: [0.9241504997544637]
---------
Input: [1.47] Expected: [0.99492437] Produced: [0.9218239694996047]
---------
Input: [0.65] Expected: [0.6051864] Produced: [0.6488632158403097]
---------
Input: [1.34] Expected: [0.9734845] Produced: [0.9090418601995752]
---------
Input: [0.02] Expected: [0.01999867] Produced: [0.08616305137994604]
---------
Input: [1.38] Expected: [0.98185354] Produced: [0.9134792089575561]
---------
Input: [0.18] Expected: [0.17902957] Produced: [0.18018325062121682]
---------
Input: [1.49] Expected: [0.9967378] Produced: [0.9234498334104785]
---------
Input: [0.07] Expected: [0.06994285] Produced: [0.10966316493438771]
---------
Input: [0.45] Expected: [0.43496552] Produced: [0.4496517253431988]
---------
Epoch 3000 RMSE = 0.053683404379706035
Epoch 3100 RMSE = 0.05279992503459655
Epoch 3200 RMSE = 0.05197837265323537
Epoch 3300 RMSE = 0.0512130973970176
Epoch 3400 RMSE = 0.05049897630871091
Epoch 3500 RMSE = 0.04983142848313984
Epoch 3600 RMSE = 0.04920644429961621
Epoch 3700 RMSE = 0.04862021608915753
Epoch 3800 RMSE = 0.048070051921190037
Epoch 3900 RMSE = 0.04755246689656714
Input: [1.47] Expected: [0.99492437] Produced: [0.9324910653297318]
---------
Input: [0.18] Expected: [0.17902957] Produced: [0.16955584035859236]
---------
Input: [0.64] Expected: [0.59719545] Produced: [0.6414887781579101]
---------
Input: [1.34] Expected: [0.9734845] Produced: [0.9198517881922044]
---------
Input: [1.49] Expected: [0.9967378] Produced: [0.9340531874983656]
---------
Input: [1.53] Expected: [0.9991679] Produced: [0.9369752258576535]
---------
Input: [0.36] Expected: [0.35227424] Produced: [0.34018696591598335]
---------
Input: [0.24] Expected: [0.23770262] Produced: [0.21899113579277296]
---------
Input: [0.02] Expected: [0.01999867] Produced: [0.07892166703994796]
---------
Input: [0.07] Expected: [0.06994285] Produced: [0.10124750791468554]
---------
Input: [1.38] Expected: [0.98185354] Produced: [0.9243267852979438]
---------
Input: [0.45] Expected: [0.43496552] Produced: [0.4422289846703868]
---------
Input: [1.5] Expected: [0.997495] Produced: [0.9348806329634725]
---------
Input: [0.65] Expected: [0.6051864] Produced: [0.6506389883135058]
---------
Input: [0.1] Expected: [0.09983341] Produced: [0.11709803302761258]
---------
Input: [0.01] Expected: [0.00999983] Produced: [0.07499048486828341]
---------
Epoch 4000 RMSE = 0.047065037413378794
Epoch 4100 RMSE = 0.04660535537726411
Epoch 4200 RMSE = 0.046171519501972035
Epoch 4300 RMSE = 0.0457610615342901
Epoch 4400 RMSE = 0.045373074477392104
Epoch 4500 RMSE = 0.04500505119564521
Epoch 4600 RMSE = 0.044655963049646634
Epoch 4700 RMSE = 0.04432425034101609
Epoch 4800 RMSE = 0.044009366437098754
Epoch 4900 RMSE = 0.043709510182374546
Input: [1.49] Expected: [0.9967378] Produced: [0.9408528219692479]
---------
Input: [0.24] Expected: [0.23770262] Produced: [0.21243175917839782]
---------
Input: [1.47] Expected: [0.99492437] Produced: [0.9393433433064331]
---------
Input: [0.1] Expected: [0.09983341] Produced: [0.11231767906911934]
---------
Input: [0.36] Expected: [0.35227424] Produced: [0.3337852516997233]
---------
Input: [1.53] Expected: [0.9991679] Produced: [0.9437822387197538]
---------
Input: [0.65] Expected: [0.6051864] Produced: [0.6511722243568422]
---------
Input: [0.18] Expected: [0.17902957] Produced: [0.16366649727946983]
---------
Input: [0.07] Expected: [0.06994285] Produced: [0.09684433252529018]
---------
Input: [0.45] Expected: [0.43496552] Produced: [0.43701609057910995]
---------
Input: [1.5] Expected: [0.997495] Produced: [0.9416236091521247]
---------
Input: [0.02] Expected: [0.01999867] Produced: [0.07519977491003033]
---------
Input: [0.01] Expected: [0.00999983] Produced: [0.07142503535653284]
---------
Input: [1.38] Expected: [0.98185354] Produced: [0.9311856518621165]
---------
Input: [0.64] Expected: [0.59719545] Produced: [0.641747584135062]
---------
Input: [1.34] Expected: [0.9734845] Produced: [0.9267829318143812]
---------
Epoch 5000 RMSE = 0.04342369652917983
Epoch 5100 RMSE = 0.04315124737730475
Epoch 5200 RMSE = 0.042891029842587726
Epoch 5300 RMSE = 0.04264263216895522
Epoch 5400 RMSE = 0.04240490925898322
Epoch 5500 RMSE = 0.04217770254856337
Epoch 5600 RMSE = 0.041959960091574336
Epoch 5700 RMSE = 0.04175099046859616
Epoch 5800 RMSE = 0.04155090882668274
Epoch 5900 RMSE = 0.0413586470089798
Input: [0.45] Expected: [0.43496552] Produced: [0.4334296095776033]
---------
Input: [1.49] Expected: [0.9967378] Produced: [0.9456082577133768]
---------
Input: [0.36] Expected: [0.35227424] Produced: [0.329477377783853]
---------
Input: [0.07] Expected: [0.06994285] Produced: [0.09441941302515951]
---------
Input: [1.5] Expected: [0.997495] Produced: [0.9463797454653672]
---------
Input: [1.38] Expected: [0.98185354] Produced: [0.9360419905838141]
---------
Input: [0.18] Expected: [0.17902957] Produced: [0.1602749311591926]
---------
Input: [0.1] Expected: [0.09983341] Produced: [0.10963431829514883]
---------
Input: [0.02] Expected: [0.01999867] Produced: [0.07321710804522973]
---------
Input: [1.47] Expected: [0.99492437] Produced: [0.9441430879324597]
---------
Input: [0.01] Expected: [0.00999983] Produced: [0.0695299947769358]
---------
Input: [0.64] Expected: [0.59719545] Produced: [0.6418071002485506]
---------
Input: [1.53] Expected: [0.9991679] Produced: [0.9484469279078148]
---------
Input: [0.24] Expected: [0.23770262] Produced: [0.208506104823279]
---------
Input: [1.34] Expected: [0.9734845] Produced: [0.9317071173598316]
---------
Input: [0.65] Expected: [0.6051864] Produced: [0.6511707914721547]
---------
Epoch 6000 RMSE = 0.04117386825546275
Epoch 6100 RMSE = 0.040995868555274415
Epoch 6200 RMSE = 0.040824988606990284
Epoch 6300 RMSE = 0.04066059789798512
Epoch 6400 RMSE = 0.040501972153336244
Epoch 6500 RMSE = 0.04034926402394175
Epoch 6600 RMSE = 0.04020163238189283
Epoch 6700 RMSE = 0.04005937582943579
Epoch 6800 RMSE = 0.03992172000947841
Epoch 6900 RMSE = 0.039788965561290204
Input: [1.5] Expected: [0.997495] Produced: [0.9498220886630808]
---------
Input: [0.24] Expected: [0.23770262] Produced: [0.2059048096065338]
---------
Input: [0.45] Expected: [0.43496552] Produced: [0.43092177668003656]
---------
Input: [1.38] Expected: [0.98185354] Produced: [0.9395544827707001]
---------
Input: [1.53] Expected: [0.9991679] Produced: [0.9519512811545962]
---------
Input: [1.34] Expected: [0.9734845] Produced: [0.9352709913651966]
---------
Input: [1.49] Expected: [0.9967378] Produced: [0.949177204707398]
---------
Input: [0.36] Expected: [0.35227424] Produced: [0.32677475978838216]
---------
Input: [0.65] Expected: [0.6051864] Produced: [0.651218120688429]
---------
Input: [0.02] Expected: [0.01999867] Produced: [0.07209034391395548]
---------
Input: [1.47] Expected: [0.99492437] Produced: [0.9476219814549453]
---------
Input: [0.01] Expected: [0.00999983] Produced: [0.06845537124141826]
---------
Input: [0.18] Expected: [0.17902957] Produced: [0.1580776707948022]
---------
Input: [0.64] Expected: [0.59719545] Produced: [0.6413933558926814]
---------
Input: [0.1] Expected: [0.09983341] Produced: [0.10795249935046265]
---------
Input: [0.07] Expected: [0.06994285] Produced: [0.09297253185080503]
---------
Epoch 7000 RMSE = 0.03966030461119905
Epoch 7100 RMSE = 0.03953602508984381
Epoch 7200 RMSE = 0.039415830109690686
Epoch 7300 RMSE = 0.03929905650312713
Epoch 7400 RMSE = 0.0391863954873961
Epoch 7500 RMSE = 0.039076919993113815
Epoch 7600 RMSE = 0.03897072580058033
Epoch 7700 RMSE = 0.03886765871787282
Epoch 7800 RMSE = 0.03876763508110721
Epoch 7900 RMSE = 0.03867041605105723
Input: [0.45] Expected: [0.43496552] Produced: [0.42876208939943455]
---------
Input: [0.65] Expected: [0.6051864] Produced: [0.6503650771804238]
---------
Input: [0.07] Expected: [0.06994285] Produced: [0.09209734175930417]
---------
Input: [1.34] Expected: [0.9734845] Produced: [0.9378303968843671]
---------
Input: [0.36] Expected: [0.35227424] Produced: [0.32410983503569374]
---------
Input: [1.47] Expected: [0.99492437] Produced: [0.950246453892088]
---------
Input: [0.1] Expected: [0.09983341] Produced: [0.10697340899698042]
---------
Input: [0.02] Expected: [0.01999867] Produced: [0.07141873901637541]
---------
Input: [1.49] Expected: [0.9967378] Produced: [0.9517689934952128]
---------
Input: [1.53] Expected: [0.9991679] Produced: [0.9545727418656228]
---------
Input: [1.38] Expected: [0.98185354] Produced: [0.9422457935216048]
---------
Input: [1.5] Expected: [0.997495] Produced: [0.9525361421630554]
---------
Input: [0.18] Expected: [0.17902957] Produced: [0.15664924723986237]
---------
Input: [0.24] Expected: [0.23770262] Produced: [0.20425708610359303]
---------
Input: [0.01] Expected: [0.00999983] Produced: [0.06786153568077928]
---------
Input: [0.64] Expected: [0.59719545] Produced: [0.6409605330967831]
---------
Epoch 8000 RMSE = 0.038576079124414485
Epoch 8100 RMSE = 0.03848432374706796
Epoch 8200 RMSE = 0.03839507259011825
Epoch 8300 RMSE = 0.03830818549155823
Epoch 8400 RMSE = 0.03822375212302759
Epoch 8500 RMSE = 0.038141550255638035
Epoch 8600 RMSE = 0.03806143839845238
Epoch 8700 RMSE = 0.03798330461378782
Epoch 8800 RMSE = 0.037906907512359166
Epoch 8900 RMSE = 0.037833072600480404
Input: [1.5] Expected: [0.997495] Produced: [0.9546488326845892]
---------
Input: [1.47] Expected: [0.99492437] Produced: [0.9524255040596539]
---------
Input: [0.02] Expected: [0.01999867] Produced: [0.07108309506857062]
---------
Input: [0.01] Expected: [0.00999983] Produced: [0.06750237319368711]
---------
Input: [1.34] Expected: [0.9734845] Produced: [0.9400958107202212]
---------
Input: [0.65] Expected: [0.6051864] Produced: [0.6500927666808222]
---------
Input: [0.45] Expected: [0.43496552] Produced: [0.42697896507332606]
---------
Input: [1.53] Expected: [0.9991679] Produced: [0.9566728695337954]
---------
Input: [0.18] Expected: [0.17902957] Produced: [0.15561608234761257]
---------
Input: [0.64] Expected: [0.59719545] Produced: [0.6401816350543239]
---------
Input: [0.36] Expected: [0.35227424] Produced: [0.32239975468369014]
---------
Input: [0.07] Expected: [0.06994285] Produced: [0.0916110585506822]
---------
Input: [1.49] Expected: [0.9967378] Produced: [0.9538816333155403]
---------
Input: [0.24] Expected: [0.23770262] Produced: [0.2028647060000577]
---------
Input: [0.1] Expected: [0.09983341] Produced: [0.10636640166475415]
---------
Input: [1.38] Expected: [0.98185354] Produced: [0.9443864355046183]
---------
Epoch 9000 RMSE = 0.037760680019334684
Epoch 9100 RMSE = 0.03769013991147963
Epoch 9200 RMSE = 0.03762134242387314
Epoch 9300 RMSE = 0.03755393706142236
Epoch 9400 RMSE = 0.03748839389014545
Epoch 9500 RMSE = 0.03742421774278871
Epoch 9600 RMSE = 0.03736143487580937
Epoch 9700 RMSE = 0.037300370683193695
Epoch 9800 RMSE = 0.03724049105819764
Epoch 9900 RMSE = 0.03718167957824631
Input: [1.49] Expected: [0.9967378] Produced: [0.955675169719665]
---------
Input: [1.34] Expected: [0.9734845] Produced: [0.9418724487686634]
---------
Input: [0.24] Expected: [0.23770262] Produced: [0.20211167164515884]
---------
Input: [0.36] Expected: [0.35227424] Produced: [0.3215714125927918]
---------
Input: [0.45] Expected: [0.43496552] Produced: [0.4262433465529339]
---------
Input: [0.18] Expected: [0.17902957] Produced: [0.15516391416693406]
---------
Input: [0.02] Expected: [0.01999867] Produced: [0.07094260144041614]
---------
Input: [0.65] Expected: [0.6051864] Produced: [0.6500939008568861]
---------
Input: [1.38] Expected: [0.98185354] Produced: [0.9462064972648122]
---------
Input: [0.64] Expected: [0.59719545] Produced: [0.6400110404524297]
---------
Input: [0.07] Expected: [0.06994285] Produced: [0.09133770581732001]
---------
Input: [0.1] Expected: [0.09983341] Produced: [0.10597837954380196]
---------
Input: [1.47] Expected: [0.99492437] Produced: [0.9541414481176168]
---------
Input: [0.01] Expected: [0.00999983] Produced: [0.06732492519721266]
---------
Input: [1.53] Expected: [0.9991679] Produced: [0.9584203599910307]
---------
Input: [1.5] Expected: [0.997495] Produced: [0.9563856561319631]
---------
Epoch 10000 RMSE = 0.03712444899081186
Final RMSE = 0.03712444899081186
---TESTING---
Input values: [0.91]
Expected output: [0.78950375]
Actual output: 0.8341501090436322 
---------
Input values: [0.53]
Expected output: [0.50553334]
Actual output: 0.5202839152003185 
---------
Input values: [1.12]
Expected output: [0.90010047]
Actual output: 0.9051289460888401 
---------
Input values: [0.77]
Expected output: [0.6961352]
Actual output: 0.7511217470739919 
---------
Input values: [0.29]
Expected output: [0.2859522]
Actual output: 0.24763535831616618 
---------
Input values: [0.31]
Expected output: [0.30505863]
Actual output: 0.26768355894634477 
---------
Input values: [0.43]
Expected output: [0.4168708]
Actual output: 0.4018046713313112 
---------
Input values: [0.55]
Expected output: [0.52268726]
Actual output: 0.5432052056131059 
---------
Input values: [1.55]
Expected output: [0.99978375]
Actual output: 0.9596367759405261 
---------
Input values: [1.03]
Expected output: [0.857299]
Actual output: 0.8805537527108608 
---------
Input values: [0.79]
Expected output: [0.71035326]
Actual output: 0.765091318326717 
---------
Input values: [0.8]
Expected output: [0.7173561]
Actual output: 0.7715524128122228 
---------
Input values: [0.54]
Expected output: [0.514136]
Actual output: 0.5309149893633911 
---------
Input values: [0.67]
Expected output: [0.620986]
Actual output: 0.6674324726411791 
---------
Input values: [0.37]
Expected output: [0.36161542]
Actual output: 0.3314121543017106 
---------
Input values: [0.89]
Expected output: [0.7770718]
Actual output: 0.8234339388448393 
---------
Input values: [1.37]
Expected output: [0.97990805]
Actual output: 0.9447365115049022 
---------
Input values: [0.06]
Expected output: [0.059964]
Actual output: 0.08667719972785434 
---------
Input values: [0.73]
Expected output: [0.66686964]
Actual output: 0.7193334713046748 
---------
Input values: [0.68]
Expected output: [0.628793]
Actual output: 0.6759529428691916 
---------
Input values: [1.09]
Expected output: [0.8866269]
Actual output: 0.8968579880059661 
---------
Input values: [1.13]
Expected output: [0.9044122]
Actual output: 0.9065947712558269 
---------
Input values: [0.82]
Expected output: [0.73114586]
Actual output: 0.7832139334800805 
---------
Input values: [0.47]
Expected output: [0.45288628]
Actual output: 0.44723489327127924 
---------
Input values: [0.99]
Expected output: [0.83602595]
Actual output: 0.8657548530604363 
---------
Input values: [1.33]
Expected output: [0.9711484]
Actual output: 0.9399618153094462 
---------
Input values: [0.59]
Expected output: [0.556361]
Actual output: 0.5851738725164805 
---------
Input values: [1.36]
Expected output: [0.9778646]
Actual output: 0.9433552635200786 
---------
Input values: [0.96]
Expected output: [0.8191916]
Actual output: 0.8541332623558298 
---------
Input values: [0.27]
Expected output: [0.26673144]
Actual output: 0.22744041131664053 
---------
Input values: [0.62]
Expected output: [0.58103514]
Actual output: 0.616700353926275 
---------
Input values: [1.02]
Expected output: [0.852108]
Actual output: 0.8758935450642513 
---------
Input values: [1.23]
Expected output: [0.9424888]
Actual output: 0.9257283956504351 
---------
Input values: [1.2]
Expected output: [0.9320391]
Actual output: 0.9205502889161309 
---------
Input values: [1.04]
Expected output: [0.8624042]
Actual output: 0.8822690631163017 
---------
Input values: [0.71]
Expected output: [0.6518338]
Actual output: 0.7012467286716525 
---------
Input values: [0.83]
Expected output: [0.7379314]
Actual output: 0.7884587103249495 
---------
Input values: [1.31]
Expected output: [0.966185]
Actual output: 0.9371824984593731 
---------
Input values: [1.35]
Expected output: [0.9757234]
Actual output: 0.9420179021570192 
---------
Input values: [0.74]
Expected output: [0.6742879]
Actual output: 0.7254001289725281 
---------
Input values: [0.97]
Expected output: [0.8248857]
Actual output: 0.8573649654741133 
---------
Input values: [0.92]
Expected output: [0.7956016]
Actual output: 0.8360078127158671 
---------
Input values: [0.72]
Expected output: [0.65938467]
Actual output: 0.708543402150118 
---------
Input values: [0.32]
Expected output: [0.31456655]
Actual output: 0.2753836401594754 
---------
Input values: [1.07]
Expected output: [0.8772005]
Actual output: 0.8904021562545312 
---------
Input values: [0.63]
Expected output: [0.58914477]
Actual output: 0.6251329897403337 
---------
Input values: [0.52]
Expected output: [0.49688014]
Actual output: 0.5040429208859845 
---------
Input values: [0.33]
Expected output: [0.32404304]
Actual output: 0.2856557172790722 
---------
Input values: [0.95]
Expected output: [0.8134155]
Actual output: 0.8488488831323231 
---------
Input values: [1.32]
Expected output: [0.9687151]
Actual output: 0.9381507814445993 
---------
Input values: [1.16]
Expected output: [0.9168031]
Actual output: 0.9121321658798979 
---------
Input values: [0.48]
Expected output: [0.46177918]
Actual output: 0.4570638356419704 
---------
Input values: [0.76]
Expected output: [0.68892145]
Actual output: 0.7398271602830178 
---------
Input values: [0.7]
Expected output: [0.64421767]
Actual output: 0.690774531868923 
---------
Input values: [1.29]
Expected output: [0.96083504]
Actual output: 0.9340618149387339 
---------
Input values: [0.85]
Expected output: [0.7512804]
Actual output: 0.7989039215585636 
---------
Input values: [0.3]
Expected output: [0.29552022]
Actual output: 0.254694575760909 
---------
Input values: [0.78]
Expected output: [0.70327944]
Actual output: 0.7538198782393638 
---------
Input values: [0.35]
Expected output: [0.3428978]
Actual output: 0.30637373568214715 
---------
Input values: [0.25]
Expected output: [0.24740396]
Actual output: 0.20826674329936304 
---------
Input values: [1.1]
Expected output: [0.89120734]
Actual output: 0.8979947126583746 
---------
Input values: [0.]
Expected output: [0.]
Actual output: 0.06354862693299773 
---------
Input values: [0.04]
Expected output: [0.03998933]
Actual output: 0.07792849136080895 
---------
Input values: [0.22]
Expected output: [0.21822962]
Actual output: 0.1832511193510595 
---------
Input values: [0.05]
Expected output: [0.04997917]
Actual output: 0.08197131708850558 
---------
Input values: [0.41]
Expected output: [0.39860934]
Actual output: 0.3740390001345733 
---------
Input values: [0.88]
Expected output: [0.7707389]
Actual output: 0.8154694610247869 
---------
Input values: [0.26]
Expected output: [0.25708055]
Actual output: 0.21710550719658897 
---------
Input values: [0.09]
Expected output: [0.08987855]
Actual output: 0.10004274740246565 
---------
Input values: [0.17]
Expected output: [0.16918235]
Actual output: 0.1465147856915134 
---------
Input values: [1.]
Expected output: [0.84147096]
Actual output: 0.867762060588267 
---------
Input values: [0.39]
Expected output: [0.3801884]
Actual output: 0.35105307703228106 
---------
Input values: [0.44]
Expected output: [0.42593947]
Actual output: 0.4092679179249249 
---------
Input values: [1.08]
Expected output: [0.8819578]
Actual output: 0.8928283616478784 
---------
Input values: [1.41]
Expected output: [0.9871001]
Actual output: 0.9478721633585657 
---------
Input values: [0.14]
Expected output: [0.13954312]
Actual output: 0.1274310094068821 
---------
Input values: [0.69]
Expected output: [0.6365372]
Actual output: 0.6818390591489786 
---------
Input values: [1.15]
Expected output: [0.91276395]
Actual output: 0.9097301930521738 
---------
Input values: [1.26]
Expected output: [0.9520903]
Actual output: 0.9297110503008696 
---------
Input values: [1.51]
Expected output: [0.9981525]
Actual output: 0.9560685951984974 
---------
Input values: [1.18]
Expected output: [0.924606]
Actual output: 0.9159513468444522 
---------
Input values: [1.42]
Expected output: [0.98865175]
Actual output: 0.9487884442832366 
---------
Input values: [0.16]
Expected output: [0.15931821]
Actual output: 0.13994676694686756 
---------
Input values: [0.4]
Expected output: [0.38941833]
Actual output: 0.36270521398379824 
---------
Input values: [0.56]
Expected output: [0.5311862]
Actual output: 0.5496025309109572 
---------
Input values: [1.17]
Expected output: [0.9207506]
Actual output: 0.9140180762931527 
---------
Input values: [1.19]
Expected output: [0.928369]
Actual output: 0.9179394098378799 
---------
Input values: [1.52]
Expected output: [0.99871016]
Actual output: 0.9568176048500797 
---------
Input values: [0.28]
Expected output: [0.27635565]
Actual output: 0.2357283478492885 
---------
Input values: [0.03]
Expected output: [0.0299955]
Actual output: 0.07415912711851548 
---------
Input values: [1.14]
Expected output: [0.9086335]
Actual output: 0.907735992498166 
---------
Input values: [1.45]
Expected output: [0.992713]
Actual output: 0.9515134915493618 
---------
Input values: [0.86]
Expected output: [0.75784254]
Actual output: 0.8049048554875318 
---------
Input values: [0.5]
Expected output: [0.47942555]
Actual output: 0.48030248914366513 
---------
Input values: [0.6]
Expected output: [0.5646425]
Actual output: 0.5933047047259579 
---------
Input values: [0.19]
Expected output: [0.1888589]
Actual output: 0.16052283471393186 
---------
Input values: [0.93]
Expected output: [0.80161995]
Actual output: 0.8397642542342308 
---------
Input values: [1.25]
Expected output: [0.9489846]
Actual output: 0.9281618648979407 
---------
Input values: [0.61]
Expected output: [0.57286745]
Actual output: 0.60358796323525 
---------
Input values: [0.2]
Expected output: [0.19866933]
Actual output: 0.16778362883247555 
---------
Input values: [0.51]
Expected output: [0.48817724]
Actual output: 0.49160896008401195 
---------
Input values: [1.11]
Expected output: [0.89569867]
Actual output: 0.9004155639772523 
---------
Input values: [1.56]
Expected output: [0.9999417]
Actual output: 0.9593404570051377 
---------
Input values: [0.42]
Expected output: [0.40776044]
Actual output: 0.3855812480504833 
---------
Input values: [0.13]
Expected output: [0.12963414]
Actual output: 0.12147347591096833 
---------
Input values: [1.06]
Expected output: [0.87235546]
Actual output: 0.8870746896916721 
---------
Input values: [1.46]
Expected output: [0.99386835]
Actual output: 0.9522357019325253 
---------
Input values: [0.34]
Expected output: [0.3334871]
Actual output: 0.29590750768361285 
---------
Input values: [0.81]
Expected output: [0.72428715]
Actual output: 0.7743947638005074 
---------
Input values: [0.12]
Expected output: [0.1197122]
Actual output: 0.1157784208230781 
---------
Input values: [1.57]
Expected output: [0.9999997]
Actual output: 0.9599681373688225 
---------
Input values: [0.84]
Expected output: [0.7446431]
Actual output: 0.7929019869828408 
---------
Input values: [1.44]
Expected output: [0.99145836]
Actual output: 0.9504862435779027 
---------
Input values: [1.48]
Expected output: [0.99588084]
Actual output: 0.9537889476724247 
---------
Input values: [0.08]
Expected output: [0.0799147]
Actual output: 0.09523030968276043 
---------
Input values: [0.58]
Expected output: [0.54802394]
Actual output: 0.5711973203140566 
---------
Input values: [0.98]
Expected output: [0.8304974]
Actual output: 0.8601269299842099 
---------
Input values: [0.21]
Expected output: [0.2084599]
Actual output: 0.17530076975833972 
---------
Input values: [0.38]
Expected output: [0.37092048]
Actual output: 0.33946638892777564 
---------
Input values: [1.22]
Expected output: [0.9390994]
Actual output: 0.9231765762453162 
---------
Input values: [1.4]
Expected output: [0.98544973]
Actual output: 0.9468130248849761 
---------
Input values: [1.01]
Expected output: [0.84683186]
Actual output: 0.8711357464425156 
---------
Input values: [1.3]
Expected output: [0.9635582]
Actual output: 0.9353310784408826 
---------
Input values: [0.57]
Expected output: [0.539632]
Actual output: 0.5602048626806692 
---------
Input values: [0.75]
Expected output: [0.6816388]
Actual output: 0.7312642266176899 
---------
Input values: [0.15]
Expected output: [0.14943813]
Actual output: 0.13338659772458542 
---------
Input values: [1.05]
Expected output: [0.86742324]
Actual output: 0.8837892015988416 
---------
Input values: [1.21]
Expected output: [0.935616]
Actual output: 0.9212702031192547 
---------
Input values: [0.46]
Expected output: [0.44394812]
Actual output: 0.43219440355817224 
---------
Input values: [0.11]
Expected output: [0.1097783]
Actual output: 0.11024240519299705 
---------
Input values: [1.24]
Expected output: [0.945784]
Actual output: 0.9264033560468535 
---------
Input values: [0.66]
Expected output: [0.61311686]
Actual output: 0.653237818517922 
---------
Input values: [0.49]
Expected output: [0.47062588]
Actual output: 0.4674789873647267 
---------
Input values: [0.94]
Expected output: [0.8075581]
Actual output: 0.8435743122110553 
---------
Input values: [1.54]
Expected output: [0.99952585]
Actual output: 0.9579330615959922 
---------
Input values: [1.27]
Expected output: [0.95510083]
Actual output: 0.9309238737602638 
---------
Input values: [0.9]
Expected output: [0.7833269]
Actual output: 0.8249565055489325 
---------
Input values: [1.43]
Expected output: [0.99010456]
Actual output: 0.9494359247634933 
---------
Input values: [0.87]
Expected output: [0.76432896]
Actual output: 0.8092667800583649 
---------
Input values: [0.23]
Expected output: [0.22797753]
Actual output: 0.19094775347520737 
---------
Input values: [1.28]
Expected output: [0.95801586]
Actual output: 0.9323042149187819 
---------
Input values: [1.39]
Expected output: [0.9837008]
Actual output: 0.9456072297496156 
---------

RMSE = 0.003196742398025066

"""

"""
---run_XOR() sample run---
---TRAINING---
Input: [0. 1.] Expected: [1.] Produced: [0.6942095899797631]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.7250276480060921]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.6517172824464361]
---------
Input: [1. 0.] Expected: [1.] Produced: [0.6880441471036419]
---------
Epoch 0 RMSE = 0.5341406031255377
Epoch 100 RMSE = 0.5031270678839276
Epoch 200 RMSE = 0.5008294292643921
Epoch 300 RMSE = 0.500663132315651
Epoch 400 RMSE = 0.5006035981265866
Epoch 500 RMSE = 0.5005454729783922
Epoch 600 RMSE = 0.500483085532924
Epoch 700 RMSE = 0.5004179805928844
Epoch 800 RMSE = 0.5003469520470479
Epoch 900 RMSE = 0.5002688929866936
Input: [0. 0.] Expected: [0.] Produced: [0.498469407288014]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.499909493760598]
---------
Input: [1. 0.] Expected: [1.] Produced: [0.5089311654485302]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.5109105806769217]
---------
Epoch 1000 RMSE = 0.5001850872797522
Epoch 1100 RMSE = 0.5000925643504401
Epoch 1200 RMSE = 0.49999013493843963
Epoch 1300 RMSE = 0.49987321958764447
Epoch 1400 RMSE = 0.499749335786127
Epoch 1500 RMSE = 0.4996076852340037
Epoch 1600 RMSE = 0.49944947998856043
Epoch 1700 RMSE = 0.49926945380604454
Epoch 1800 RMSE = 0.4990679548074208
Epoch 1900 RMSE = 0.4988339854577462
Input: [1. 0.] Expected: [1.] Produced: [0.5129345430528278]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.49340476104085496]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.515379892518555]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.5019862791390991]
---------
Epoch 2000 RMSE = 0.49857675370676313
Epoch 2100 RMSE = 0.49827678805038655
Epoch 2200 RMSE = 0.49794102604896545
Epoch 2300 RMSE = 0.4975507940733269
Epoch 2400 RMSE = 0.4971099272961116
Epoch 2500 RMSE = 0.49660690233471144
Epoch 2600 RMSE = 0.496037675599169
Epoch 2700 RMSE = 0.49538473748748285
Epoch 2800 RMSE = 0.49464158839329786
Epoch 2900 RMSE = 0.4938034156712986
Input: [1. 0.] Expected: [1.] Produced: [0.526477113597183]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.5137835130427403]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.5311973751197402]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.4783414443359661]
---------
Epoch 3000 RMSE = 0.49285179943869395
Epoch 3100 RMSE = 0.4917929882340054
Epoch 3200 RMSE = 0.4906064651370485
Epoch 3300 RMSE = 0.4892878477617859
Epoch 3400 RMSE = 0.48783038484973784
Epoch 3500 RMSE = 0.4862346106743326
Epoch 3600 RMSE = 0.48449049963198204
Epoch 3700 RMSE = 0.48260795130211365
Epoch 3800 RMSE = 0.4805927588111119
Epoch 3900 RMSE = 0.4784363159145356
Input: [0. 1.] Expected: [1.] Produced: [0.5298561605948288]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.5353403560064398]
---------
Input: [1. 0.] Expected: [1.] Produced: [0.5540474721425566]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.4476342097419076]
---------
Epoch 4000 RMSE = 0.47614981080891
Epoch 4100 RMSE = 0.47373603268483094
Epoch 4200 RMSE = 0.4712094920846877
Epoch 4300 RMSE = 0.46857715248567106
Epoch 4400 RMSE = 0.46585222099158635
Epoch 4500 RMSE = 0.4630260389934951
Epoch 4600 RMSE = 0.4601188393591959
Epoch 4700 RMSE = 0.45713743247001215
Epoch 4800 RMSE = 0.45409131819588655
Epoch 4900 RMSE = 0.45097343778339094
Input: [1. 0.] Expected: [1.] Produced: [0.587131025673691]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.5186180991986362]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.4046959662897153]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.5540485902806355]
---------
Epoch 5000 RMSE = 0.44779376049123326
Epoch 5100 RMSE = 0.4445434214937907
Epoch 5200 RMSE = 0.4412420247489282
Epoch 5300 RMSE = 0.4378737256357287
Epoch 5400 RMSE = 0.43443439980792625
Epoch 5500 RMSE = 0.43091939862574596
Epoch 5600 RMSE = 0.427320185459214
Epoch 5700 RMSE = 0.4236175575418352
Epoch 5800 RMSE = 0.4198018860350795
Epoch 5900 RMSE = 0.41585642428085057
Input: [0. 1.] Expected: [1.] Produced: [0.5929877685292434]
---------
Input: [1. 0.] Expected: [1.] Produced: [0.6181538623376784]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.4832820295751685]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.3648478796491231]
---------
Epoch 6000 RMSE = 0.41174656185850306
Epoch 6100 RMSE = 0.40746358472932226
Epoch 6200 RMSE = 0.40297409727085776
Epoch 6300 RMSE = 0.39823136687250743
Epoch 6400 RMSE = 0.3932250173294906
Epoch 6500 RMSE = 0.3879065190319619
Epoch 6600 RMSE = 0.38226835036342993
Epoch 6700 RMSE = 0.3762729488507414
Epoch 6800 RMSE = 0.36990264634383857
Epoch 6900 RMSE = 0.36318285099436276
Input: [1. 1.] Expected: [0.] Produced: [0.40088888984728804]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.6441156837699101]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.3207529727667787]
---------
Input: [1. 0.] Expected: [1.] Produced: [0.6579719282525137]
---------
Epoch 7000 RMSE = 0.3561008354186982
Epoch 7100 RMSE = 0.34870001837204045
Epoch 7200 RMSE = 0.3410182610719368
Epoch 7300 RMSE = 0.3331233187479956
Epoch 7400 RMSE = 0.32508168132447407
Epoch 7500 RMSE = 0.31695067175167946
Epoch 7600 RMSE = 0.3088234967416704
Epoch 7700 RMSE = 0.3007532067204124
Epoch 7800 RMSE = 0.2928081663305052
Epoch 7900 RMSE = 0.2850452204657868
Input: [0. 0.] Expected: [0.] Produced: [0.27487824227014884]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.2877507529180387]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.7230997191268879]
---------
Input: [1. 0.] Expected: [1.] Produced: [0.7298521898963602]
---------
Epoch 8000 RMSE = 0.2774942110791797
Epoch 8100 RMSE = 0.27020155934233686
Epoch 8200 RMSE = 0.2631787102305335
Epoch 8300 RMSE = 0.2564403876938586
Epoch 8400 RMSE = 0.24999443784832429
Epoch 8500 RMSE = 0.24384454657114923
Epoch 8600 RMSE = 0.2379799722536296
Epoch 8700 RMSE = 0.23239555054756392
Epoch 8800 RMSE = 0.22708173387118158
Epoch 8900 RMSE = 0.2220297315578792
Input: [0. 1.] Expected: [1.] Produced: [0.7871613627945899]
---------
Input: [1. 0.] Expected: [1.] Produced: [0.7898907470934856]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.23791875222156852]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.2066190715380448]
---------
Epoch 9000 RMSE = 0.21722278714852958
Epoch 9100 RMSE = 0.21265267884970077
Epoch 9200 RMSE = 0.20830289758017728
Epoch 9300 RMSE = 0.20416262621849743
Epoch 9400 RMSE = 0.20021875002399295
Epoch 9500 RMSE = 0.19645999624434232
Epoch 9600 RMSE = 0.1928757012722057
Epoch 9700 RMSE = 0.1894550168470655
Epoch 9800 RMSE = 0.18618698202118628
Epoch 9900 RMSE = 0.18306252043116286
Input: [1. 0.] Expected: [1.] Produced: [0.8265600186828771]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.8257421661924139]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.15784083552592218]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.21058247951409076]
---------
Epoch 10000 RMSE = 0.18007354733581768
Epoch 10100 RMSE = 0.17721164660409622
Epoch 10200 RMSE = 0.17446818189006602
Epoch 10300 RMSE = 0.17183713688346797
Epoch 10400 RMSE = 0.16931188235549055
Epoch 10500 RMSE = 0.16688531207455165
Epoch 10600 RMSE = 0.16455249453458093
Epoch 10700 RMSE = 0.16230769357700342
Epoch 10800 RMSE = 0.1601457529796662
Epoch 10900 RMSE = 0.15806290206896192
Input: [1. 1.] Expected: [0.] Produced: [0.1270241817321713]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.19059835725647578]
---------
Input: [1. 0.] Expected: [1.] Produced: [0.8502855504910918]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.8498866584063774]
---------
Epoch 11000 RMSE = 0.15605392357271403
Epoch 11100 RMSE = 0.15411527011965673
Epoch 11200 RMSE = 0.1522432304480827
Epoch 11300 RMSE = 0.15043391467214406
Epoch 11400 RMSE = 0.14868472550231898
Epoch 11500 RMSE = 0.1469920303969269
Epoch 11600 RMSE = 0.14535325875195088
Epoch 11700 RMSE = 0.1437656700751529
Epoch 11800 RMSE = 0.1422267884559593
Epoch 11900 RMSE = 0.1407342901489613
Input: [1. 0.] Expected: [1.] Produced: [0.8675192200625601]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.17585255749141612]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.10697021316606144]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.8670172518245317]
---------
Epoch 12000 RMSE = 0.13928596200460838
Epoch 12100 RMSE = 0.13787984131734052
Epoch 12200 RMSE = 0.13651396909272417
Epoch 12300 RMSE = 0.13518650167648336
Epoch 12400 RMSE = 0.13389561608648778
Epoch 12500 RMSE = 0.1326399675168364
Epoch 12600 RMSE = 0.13141800278878707
Epoch 12700 RMSE = 0.13022809997608314
Epoch 12800 RMSE = 0.12906901242861876
Epoch 12900 RMSE = 0.1279393889431697
Input: [0. 1.] Expected: [1.] Produced: [0.8797168387467245]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.16385517185787543]
---------
Input: [1. 0.] Expected: [1.] Produced: [0.8798804929480206]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.09277153860878797]
---------
Epoch 13000 RMSE = 0.12683829328707058
Epoch 13100 RMSE = 0.1257643168930694
Epoch 13200 RMSE = 0.1247164957737344
Epoch 13300 RMSE = 0.12369379658487883
Epoch 13400 RMSE = 0.12269524159585884
Epoch 13500 RMSE = 0.1217199298190767
Epoch 13600 RMSE = 0.12076694483601243
Epoch 13700 RMSE = 0.11983564084412493
Epoch 13800 RMSE = 0.11892493391973805
Epoch 13900 RMSE = 0.11803435599666887
Input: [0. 0.] Expected: [0.] Produced: [0.1539518141585157]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.8892717254803993]
---------
Input: [1. 0.] Expected: [1.] Produced: [0.889475618698399]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.08204430076221401]
---------
Epoch 14000 RMSE = 0.11716315334870486
Epoch 14100 RMSE = 0.11631049924803954
Epoch 14200 RMSE = 0.1154758277911847
Epoch 14300 RMSE = 0.1146585368566264
Epoch 14400 RMSE = 0.11385800976144887
Epoch 14500 RMSE = 0.11307373113619586
Epoch 14600 RMSE = 0.11230517866244183
Epoch 14700 RMSE = 0.11155181435210422
Epoch 14800 RMSE = 0.11081313126666179
Epoch 14900 RMSE = 0.11008865758518663
Input: [0. 0.] Expected: [0.] Produced: [0.14572139607152546]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.8970047630704859]
---------
Input: [1. 0.] Expected: [1.] Produced: [0.8971606306724429]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.07372615966352054]
---------
Epoch 15000 RMSE = 0.10937804464976249
Epoch 15100 RMSE = 0.10868075406181667
Epoch 15200 RMSE = 0.10799638377742704
Epoch 15300 RMSE = 0.10732464123555578
Epoch 15400 RMSE = 0.10666505541292672
Epoch 15500 RMSE = 0.10601727849453156
Epoch 15600 RMSE = 0.10538100716032674
Epoch 15700 RMSE = 0.10475585647049147
Epoch 15800 RMSE = 0.10414160500813406
Epoch 15900 RMSE = 0.10353782612481778
Input: [0. 0.] Expected: [0.] Produced: [0.13868096850352252]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.9033492110638925]
---------
Input: [1. 0.] Expected: [1.] Produced: [0.9034729323230826]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.06707340976131908]
---------
Epoch 16000 RMSE = 0.10294428483208891
Epoch 16100 RMSE = 0.10236069012390264
Epoch 16200 RMSE = 0.10178676604620893
Epoch 16300 RMSE = 0.10122224018262228
Epoch 16400 RMSE = 0.10066690675300556
Epoch 16500 RMSE = 0.10012047899136324
Epoch 16600 RMSE = 0.09958272461647671
Epoch 16700 RMSE = 0.09905342784582429
Epoch 16800 RMSE = 0.09853236835835566
Epoch 16900 RMSE = 0.09801931208415995
Input: [1. 1.] Expected: [0.] Produced: [0.06159974884532694]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.9087090478276476]
---------
Input: [1. 0.] Expected: [1.] Produced: [0.9088081686768072]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.13263297192702744]
---------
Epoch 17000 RMSE = 0.09751413019226415
Epoch 17100 RMSE = 0.09701654843038121
Epoch 17200 RMSE = 0.09652641125512727
Epoch 17300 RMSE = 0.09604352601892503
Epoch 17400 RMSE = 0.09556773940567928
Epoch 17500 RMSE = 0.0950988412995207
Epoch 17600 RMSE = 0.0946367090272755
Epoch 17700 RMSE = 0.09418116058677652
Epoch 17800 RMSE = 0.093732019094354
Epoch 17900 RMSE = 0.09328921142960456
Input: [1. 0.] Expected: [1.] Produced: [0.9133364924523969]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.913312938662376]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.12728499939984694]
---------
Input: [1. 1.] Expected: [0.] Produced: [0.05709346857323282]
---------
Epoch 18000 RMSE = 0.092852497771665
Epoch 18100 RMSE = 0.09242181618235266
Epoch 18200 RMSE = 0.09199697680149622
Epoch 18300 RMSE = 0.09157788338085125
Epoch 18400 RMSE = 0.09116439655756549
Epoch 18500 RMSE = 0.0907563833253137
Epoch 18600 RMSE = 0.09035372337587727
Epoch 18700 RMSE = 0.0899563252662368
Epoch 18800 RMSE = 0.08956405041633675
Epoch 18900 RMSE = 0.08917680739823183
Input: [1. 1.] Expected: [0.] Produced: [0.05322514556715897]
---------
Input: [0. 1.] Expected: [1.] Produced: [0.9171975103148103]
---------
Input: [0. 0.] Expected: [0.] Produced: [0.12245614818566132]
---------
Input: [1. 0.] Expected: [1.] Produced: [0.9172161549029628]
---------
Epoch 19000 RMSE = 0.08879448414207677
Epoch 19100 RMSE = 0.0884169674710756
Epoch 19200 RMSE = 0.08804416528458507
Epoch 19300 RMSE = 0.08767598732106853
Epoch 19400 RMSE = 0.08731235118561274
Epoch 19500 RMSE = 0.0869531314310493
Epoch 19600 RMSE = 0.08659824062561124
Epoch 19700 RMSE = 0.08624763971836095
Epoch 19800 RMSE = 0.08590119844807712
Epoch 19900 RMSE = 0.08555882945036454
Final RMSE = 0.08555882945036454

"""