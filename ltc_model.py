import tensorflow as tf
import numpy as np
import time
import os
from enum import Enum

class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2

class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

class LTCCell(tf.keras.layers.Layer):

    def __init__(self, num_units):
        super(LTCCell, self).__init__()
        self._input_size = -1
        self._num_units = num_units
        self._is_built = False
        self._ode_solver_unfolds = 3
        self._solver = ODESolver.SemiImplicit
        self._input_mapping = MappingType.Affine
        self._erev_init_factor = 1
        self._w_init_max = 1.0
        self._w_init_min = 0.01
        self._cm_init_min = 0.5
        self._cm_init_max = 0.5
        self._gleak_init_min = 1
        self._gleak_init_max = 1
        self._w_min_value = 0.001
        self._w_max_value = 10.0
        self._gleak_min_value = 0.001
        self._gleak_max_value = 10.0
        self._cm_t_min_value = 0.001
        self._cm_t_max_value = 10.0
        self._fix_cm = None
        self._fix_gleak = None
        self._fix_vleak = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        self._input_size = input_shape[-1]
        self.sensory_mu = self.add_weight(name='sensory_mu', shape=[self._input_size, self._num_units], trainable=True, initializer=tf.random_uniform_initializer(minval=0.4, maxval=0.6))
        self.sensory_sigma = self.add_weight(name='sensory_sigma', shape=[self._input_size, self._num_units], trainable=True, initializer=tf.random_uniform_initializer(minval=1.0, maxval=2.0))
        self.sensory_W = self.add_weight(name='sensory_W', shape=[self._input_size, self._num_units], trainable=True, initializer=tf.constant_initializer(np.random.uniform(low=self._w_init_min, high=self._w_init_max, size=[self._input_size, self._num_units])))
        sensory_erev_init = 2 * np.random.randint(low=0, high=2, size=[self._input_size, self._num_units]) - 1
        self.sensory_erev = self.add_weight(name='sensory_erev', shape=[self._input_size, self._num_units], trainable=True, initializer=tf.constant_initializer(sensory_erev_init * self._erev_init_factor))
        self.mu = self.add_weight(name='mu', shape=[self._num_units, self._num_units], trainable=True, initializer=tf.random_uniform_initializer(minval=0.4, maxval=0.6))
        self.sigma = self.add_weight(name='sigma', shape=[self._num_units, self._num_units], trainable=True, initializer=tf.random_uniform_initializer(minval=1.0, maxval=2.0))
        self.W = self.add_weight(name='W', shape=[self._num_units, self._num_units], trainable=True, initializer=tf.constant_initializer(np.random.uniform(low=self._w_init_min, high=self._w_init_max, size=[self._num_units, self._num_units])))
        erev_init = 2 * np.random.randint(low=0, high=2, size=[self._num_units, self._num_units]) - 1
        self.erev = self.add_weight(name='erev', shape=[self._num_units, self._num_units], trainable=True, initializer=tf.constant_initializer(erev_init * self._erev_init_factor))
        if self._fix_vleak is None:
            self.vleak = self.add_weight(name='vleak', shape=[self._num_units], trainable=True, initializer=tf.random_uniform_initializer(minval=-0.2, maxval=0.2))
        else:
            self.vleak = self.add_weight(name='vleak', shape=[self._num_units], trainable=False, initializer=tf.constant_initializer(self._fix_vleak))
        if self._fix_gleak is None:
            initializer = tf.constant_initializer(self._gleak_init_min)
            if self._gleak_init_max > self._gleak_init_min:
                initializer = tf.random_uniform_initializer(minval=self._gleak_init_min, maxval=self._gleak_init_max)
            self.gleak = self.add_weight(name='gleak', shape=[self._num_units], trainable=True, initializer=initializer)
        else:
            self.gleak = self.add_weight(name='gleak', shape=[self._num_units], trainable=False, initializer=tf.constant_initializer(self._fix_gleak))
        if self._fix_cm is None:
            initializer = tf.constant_initializer(self._cm_init_min)
            if self._cm_init_max > self._cm_init_min:
                initializer = tf.random_uniform_initializer(minval=self._cm_init_min, maxval=self._cm_init_max)
            self.cm_t = self.add_weight(name='cm_t', shape=[self._num_units], trainable=True, initializer=initializer)
        else:
            self.cm_t = self.add_weight(name='cm_t', shape=[self._num_units], trainable=False, initializer=tf.constant_initializer(self._fix_cm))

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = tf.reshape(v_pre, [-1, v_pre.shape[-1], 1])
        mues = v_pre - mu
        x = sigma * mues
        return tf.nn.sigmoid(x)
    
    def _map_inputs(self, inputs):
        # Apply the mapping based on MappingType
        if self._input_mapping == MappingType.Identity:
            return inputs
        # You can implement other mapping types here (Affine or Linear)
        return inputs

    def call(self, inputs, states):
        state = states[0]
        inputs = self._map_inputs(inputs)
        if self._solver == ODESolver.Explicit:
            next_state = self._ode_step_explicit(inputs, state, _ode_solver_unfolds=self._ode_solver_unfolds)
        elif self._solver == ODESolver.SemiImplicit:
            next_state = self._ode_step(inputs, state)
        elif self._solver == ODESolver.RungeKutta:
            next_state = self._ode_step_runge_kutta(inputs, state)
        else:
            raise ValueError("Unknown ODE solver '{}'".format(str(self._solver)))
        outputs = next_state
        return outputs, [next_state]

    def _ode_step(self, inputs, state):
        v_pre = state
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_rev_activation = sensory_w_activation * self.sensory_erev
        w_numerator_sensory = tf.reduce_sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = tf.reduce_sum(sensory_w_activation, axis=1)
        for t in range(self._ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            rev_activation = w_activation * self.erev
            w_numerator = tf.reduce_sum(rev_activation, axis=1) + w_numerator_sensory
            w_denominator = tf.reduce_sum(w_activation, axis=1) + w_denominator_sensory
            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator
            v_pre = numerator / (denominator + 1e-8)
        return v_pre

    def _ode_step_runge_kutta(self, inputs, state):
        h = 0.1
        for i in range(self._ode_solver_unfolds):
            k1 = h * self._f_prime(inputs, state)
            k2 = h * self._f_prime(inputs, state + k1 * 0.5)
            k3 = h * self._f_prime(inputs, state + k2 * 0.5)
            k4 = h * self._f_prime(inputs, state + k3)
            state = state + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return state

    def _ode_step_explicit(self, inputs, state, _ode_solver_unfolds):
        v_pre = state
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        w_reduced_sensory = tf.reduce_sum(sensory_w_activation, axis=1)
        for t in range(_ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            w_reduced_synapse = tf.reduce_sum(w_activation, axis=1)
            sensory_in = self.sensory_erev * sensory_w_activation
            synapse_in = self.erev * w_activation
            sum_in = tf.reduce_sum(sensory_in, axis=1) - v_pre * w_reduced_synapse + tf.reduce_sum(synapse_in, axis=1) - v_pre * w_reduced_sensory
            f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)
            v_pre = v_pre + 0.1 * f_prime
        return v_pre

    def _f_prime(self, inputs, state):
        v_pre = state
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        w_reduced_sensory = tf.reduce_sum(sensory_w_activation, axis=1)
        w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
        w_reduced_synapse = tf.reduce_sum(w_activation, axis=1)
        sensory_in = self.sensory_erev * sensory_w_activation
        synapse_in = self.erev * w_activation
        sum_in = tf.reduce_sum(sensory_in, axis=1) - v_pre * w_reduced_synapse + tf.reduce_sum(synapse_in, axis=1) - v_pre * w_reduced_sensory
        f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)
        return f_prime

    def get_param_constrain_op(self):
        cm_clipping_op = tf.clip_by_value(self.cm_t, self._cm_t_min_value, self._cm_t_max_value)
        gleak_clipping_op = tf.clip_by_value(self.gleak, self._gleak_min_value, self._gleak_max_value)
        w_clipping_op = tf.clip_by_value(self.W, self._w_min_value, self._w_max_value)
        sensory_w_clipping_op = tf.clip_by_value(self.sensory_W, self._w_min_value, self._w_max_value)
        return [cm_clipping_op, gleak_clipping_op, w_clipping_op, sensory_w_clipping_op]

    def export_weights(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        weights = [self.W, self.erev, self.mu, self.sigma, self.sensory_W, self.sensory_erev, self.sensory_mu, self.sensory_sigma, self.vleak, self.gleak, self.cm_t]
        weight_names = ["w", "erev", "mu", "sigma", "sensory_w", "sensory_erev", "sensory_mu", "sensory_sigma", "vleak", "gleak", "cm"]
        for weight, name in zip(weights, weight_names):
            np.savetxt(os.path.join(dirname, f"{name}.csv"), weight.numpy())