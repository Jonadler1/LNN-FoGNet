import tensorflow as tf
import numpy as np
import os

class CTRNN(tf.keras.layers.Layer):

    def __init__(self, num_units, cell_clip=-1, global_feedback=False, fix_tau=True):
        super(CTRNN, self).__init__()
        self._num_units = num_units
        self._unfolds = 6
        self._delta_t = 0.1
        self.global_feedback = global_feedback
        self.fix_tau = fix_tau
        self.tau = 1
        self.cell_clip = cell_clip

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        input_size = input_shape[-1]
        state_size = self._num_units

        # Update W to account for both input and state concatenation
        self.W = self.add_weight(shape=(input_size + state_size, self._num_units), initializer='glorot_uniform', name='W')
        self.b = self.add_weight(shape=(self._num_units,), initializer='zeros', name='b')
        if not self.fix_tau:
            self._tau_var = self.add_weight(shape=(), initializer=tf.constant_initializer(self.tau), name='tau')

    def _dense(self, inputs, activation=None):
        y = tf.matmul(inputs, self.W) + self.b
        if activation is not None:
            y = activation(y)
        return y

    def call(self, inputs, states):
        state = states[0]
        if not self.fix_tau:
            tau = tf.nn.softplus(self._tau_var)
        else:
            tau = self.tau

        for i in range(self._unfolds):
            # Concatenate inputs and state for calculating f_prime
            fused_input = tf.concat([inputs, state], axis=-1)
            input_f_prime = self._dense(fused_input, activation=tf.nn.tanh)

            f_prime = -state / tau + input_f_prime
            state = state + self._delta_t * f_prime

            if self.cell_clip > 0:
                state = tf.clip_by_value(state, -self.cell_clip, self.cell_clip)

        return state, [state]


class NODE(tf.keras.layers.Layer):

    def __init__(self, num_units, cell_clip=-1):
        super(NODE, self).__init__()
        self._num_units = num_units
        self._unfolds = 6
        self._delta_t = 0.1
        self.cell_clip = cell_clip

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        input_size = input_shape[-1]
        self.W = self.add_weight(shape=(input_size + self._num_units, self._num_units), initializer='glorot_uniform', name='W')
        self.b = self.add_weight(shape=(self._num_units,), initializer='zeros', name='b')

    def _dense(self, inputs, activation=None):
        y = tf.matmul(inputs, self.W) + self.b
        if activation is not None:
            y = activation(y)
        return y

    def _ode_step_runge_kutta(self, inputs, state):
        for i in range(self._unfolds):
            k1 = self._delta_t * self._f_prime(inputs, state)
            k2 = self._delta_t * self._f_prime(inputs, state + k1 * 0.5)
            k3 = self._delta_t * self._f_prime(inputs, state + k2 * 0.5)
            k4 = self._delta_t * self._f_prime(inputs, state + k3)

            state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

            if self.cell_clip > 0:
                state = tf.clip_by_value(state, -self.cell_clip, self.cell_clip)

        return state

    def _f_prime(self, inputs, state):
        fused_input = tf.concat([inputs, state], axis=-1)
        input_f_prime = self._dense(fused_input, activation=tf.nn.tanh)
        return input_f_prime

    def call(self, inputs, states):
        state = states[0]
        state = self._ode_step_runge_kutta(inputs, state)
        return state, [state]


class CTGRU(tf.keras.layers.Layer):
    def __init__(self, num_units, M=8, cell_clip=-1):
        super(CTGRU, self).__init__()
        self._num_units = num_units
        self.M = M
        self.cell_clip = cell_clip
        self.ln_tau_table = np.empty(self.M)
        tau = 1
        for i in range(self.M):
            self.ln_tau_table[i] = np.log(tau)
            tau = tau * (10.0 ** 0.5)

    @property
    def state_size(self):
        return self._num_units * self.M

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        input_size = input_shape[-1]
        self.W = self.add_weight(shape=(input_size + self._num_units, self._num_units), initializer='glorot_uniform', name='W')
        self.b = self.add_weight(shape=(self._num_units,), initializer='zeros', name='b')

    def _dense(self, inputs, activation=None):
        y = tf.matmul(inputs, self.W) + self.b
        if activation is not None:
            y = activation(y)
        return y

    def call(self, inputs, states):
        state = states[0]
        h_hat = tf.reshape(state, [-1, self._num_units, self.M])
        h = tf.reduce_sum(h_hat, axis=2)

        fused_input = tf.concat([inputs, h], axis=-1)
        ln_tau_r = tf.keras.layers.Dense(self._num_units * self.M, activation=None)(fused_input)
        ln_tau_r = tf.reshape(ln_tau_r, shape=[-1, self._num_units, self.M])
        sf_input_r = -tf.square(ln_tau_r - self.ln_tau_table)
        rki = tf.nn.softmax(sf_input_r, axis=2)

        q_input = tf.reduce_sum(rki * h_hat, axis=2)
        reset_value = tf.concat([inputs, q_input], axis=-1)
        qk = self._dense(reset_value, activation=tf.nn.tanh)

        qk = tf.reshape(qk, [-1, self._num_units, 1])

        ln_tau_s = tf.keras.layers.Dense(self._num_units * self.M, activation=None)(fused_input)
        ln_tau_s = tf.reshape(ln_tau_s, shape=[-1, self._num_units, self.M])
        sf_input_s = -tf.square(ln_tau_s - self.ln_tau_table)
        ski = tf.nn.softmax(sf_input_s, axis=2)

        h_hat_next = ((1 - ski) * h_hat + ski * qk) * np.exp(-1.0 / self.ln_tau_table)

        if self.cell_clip > 0:
            h_hat_next = tf.clip_by_value(h_hat_next, -self.cell_clip, self.cell_clip)

        h_next = tf.reduce_sum(h_hat_next, axis=2)
        h_hat_next_flat = tf.reshape(h_hat_next, shape=[-1, self._num_units * self.M])

        return h_next, [h_hat_next_flat]