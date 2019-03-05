#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daw two-step task

"""
from __future__ import division

import numpy as np
from gym import spaces
import tasktools
import ngym


class DawTwoStep(ngym.ngym):
    # TODO: remain to be carefully tested
    # Inputs
    inputs = tasktools.to_map('FIXATION', 'STATE1', 'STATE2')

    # Actions
    actions = tasktools.to_map('FIXATE', 'ACTION1', 'ACTION2')

    # trial conditions
    p1 = 0.8  # first stage transition probability
    p2 = 0.8  # second stage transition probability
    # TODO: this should be implemented through the wrapper
    p_switch = 0.025  # switch reward contingency
    high_reward_p = 0.9
    low_reward_p = 0.1
    state1_high_reward = True
    tmax = 2

    # Input noise
    sigma = np.sqrt(2*100*0.01)

    # Rewards
    R_ABORTED = -1.
    R_CORRECT = +1.
    R_FAIL = 0.
    R_MISS = 0.
    abort = False

    def __init__(self, dt=100):
        super().__init__(dt=dt)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.trial = self._new_trial()

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        # determine the transitions
        transition = dict()
        st1 = self.inputs['STATE1']
        st2 = self.inputs['STATE2']
        tmp1 = st1 if self.rng.rand() < self.p1 else st2
        tmp2 = st2 if self.rng.rand() < self.p1 else st1
        transition[self.actions['ACTION1']] = tmp1
        transition[self.actions['ACTION2']] = tmp2

        if self.state1_high_reward:
            hi_state, low_state = 'STATE1', 'STATE2'
        else:
            hi_state, low_state = 'STATE2', 'STATE1'

        reward = dict()
        reward[self.inputs[hi_state]] =\
            (self.rng.rand() < self.high_reward_p) * self.R_CORRECT
        reward[self.inputs[low_state]] =\
            (self.rng.rand() < self.low_reward_p) * self.R_CORRECT

        return {
            'transition':  transition,
            'reward': reward
            }

    def _step(self, action):
        trial = self.trial
        info = {'continue': True}
        reward = 0
        tr_perf = False  # TODO: never set to True

        # TODO: should we section by reward/input or epochs?
        obs = np.zeros(len(self.inputs))
        if self.t == 0:  # at stage 1
            if action == self.actions['FIXATE']:
                reward = self.R_ABORTED
                info['continue'] = not self.abort
            else:
                state = trial['transition'][action]
                obs[state] = 1
                reward = trial['reward'][state]
        elif self.t == 1:
            obs[self.inputs['FIXATION']] = 1
            if action != self.actions['FIXATE']:
                reward = self.R_ABORTED
            else:
                tr_perf = True
            info['continue'] = False
        else:
            raise ValueError('t is not 0 or 1')

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax, self.dt,
                                                info['continue'],
                                                self.R_MISS, reward)

        if new_trial:
            info['new_trial'] = True
            info['gt'] = trial['ground_truth']
            self.t = 0
            self.num_tr += 1
            # compute perf
            self.perf, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward,
                                       self.num_tr_perf, tr_perf)
            self.trial = self._new_trial()
        else:
            self.t += 1

        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info, new_trial

    def step(self, action):
        obs, reward, done, info, new_trial = self._step(action)
        if new_trial:
            self.trial = self._new_trial()
        return obs, reward, done, info

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.8