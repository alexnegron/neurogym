#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:10:35 2019

@author: molano

Economic choice task, based on

  Neurons in the orbitofrontal cortex encode economic value.
  C Padoa-Schioppa & JA Assad, Nature 2006.

  http://dx.doi.org/10.1038/nature04676

"""
from __future__ import division

import numpy as np
import ngym
from gym import spaces
import tasktools


class PadoaSch(ngym.ngym):
    # Inputs
    inputs = tasktools.to_map('FIXATION', 'L-A', 'L-B', 'R-A',
                              'R-B', 'N-L', 'N-R')

    # Actions
    actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT')

    # Trial conditions
    A_to_B = 2.2
    juices = [('A', 'B'), ('B', 'A')]
    offers = [(0, 1), (1, 3), (1, 2), (1, 1), (2, 1),
              (3, 1), (4, 1), (6, 1), (2, 0)]

    # Input noise
    sigma = np.sqrt(2*100*0.001)

    # Durations
    fixation = 750
    offer_on_min = 1000
    offer_on_max = 2000
    decision = 750
    tmax = fixation + offer_on_max + decision

    # Rewards
    R_ABORTED = -1.
    R_MISS = 0.
    R_B = 0.1
    R_A = A_to_B * R_B

    # Increase initial policy -> baseline weights
    baseline_Win = 10

    def __init__(self, dt=50):
        # call ngm __init__ function
        super().__init__(dt=dt)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7, ),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.steps_beyond_done = None

        self.trial = self._new_trial(self.rng, self.dt)

    # Input scaling
    def scale(self, x):
        return x/5

    def _new_trial(self, rng, dt, context={}):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------

        offer_on = context.get('offer-on')
        if offer_on is None:
            offer_on = tasktools.uniform(rng, dt, self.offer_on_min,
                                         self.offer_on_max)

        durations = {
            'fixation':    (0, self.fixation),
            'offer-on':    (self.fixation, self.fixation + offer_on),
            'decision':    (self.fixation + offer_on, self.tmax),
            'tmax':        self.tmax
            }
        time, epochs = tasktools.get_epochs_idx(dt, durations)

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        juice = context.get('juice')
        if juice is None:
            juice = tasktools.choice(rng, self.juices)

        offer = context.get('offer')
        if offer is None:
            offer = tasktools.choice(rng, self.offers)

        juiceL, juiceR = juice
        nB, nA = offer

        if juiceL == 'A':
            nL, nR = nA, nB
        else:
            nL, nR = nB, nA

        return {
            'durations': durations,
            'time':      time,
            'epochs':    epochs,
            'juice':     juice,
            'offer':     offer,
            'nL':        nL,
            'nR':        nR
            }

    def step(self, action):
        trial = self.trial

        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------

        # epochs = trial['epochs']
        info = {'continue': True}
        reward = 0
        tr_perf = False
        if (self.in_epoch(self.t, 'fixation') or
                self.in_epoch(self.t, 'offer-on')):
            if action != self.actions['FIXATE']:
                info['continue'] = False
                reward = self.R_ABORTED
        elif self.in_epoch(self.t, 'decision'):
            if action in [self.actions['CHOOSE-LEFT'],
                          self.actions['CHOOSE-RIGHT']]:
                tr_perf = True
                info['continue'] = False
                info['t_choice'] = self.t

                juiceL, juiceR = trial['juice']

                nB, nA = trial['offer']
                rA = nA * self.R_A
                rB = nB * self.R_B

                if juiceL == 'A':
                    rL, rR = rA, rB
                else:
                    rL, rR = rB, rA

                if action == self.actions['CHOOSE-LEFT']:
                    if juiceL == 'A':
                        info['choice'] = 'A'
                    else:
                        info['choice'] = 'B'
                    info['correct'] = (rL >= rR)
                    reward = rL
                elif action == self.actions['CHOOSE-RIGHT']:
                    if juiceR == 'A':
                        info['choice'] = 'A'
                    else:
                        info['choice'] = 'B'
                    info['correct'] = (rR >= rL)
                    reward = rR

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------
        obs = np.zeros(len(self.inputs))
        if not self.in_epoch(self.t, 'decision'):
            obs[self.inputs['FIXATION']] = 1
        if self.in_epoch(self.t, 'offer-on'):
            juiceL, juiceR = trial['juice']
            obs[self.inputs['L-'+juiceL]] = 1
            obs[self.inputs['R-'+juiceR]] = 1

            obs[self.inputs['N-L']] = self.scale(trial['nL']) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
            obs[self.inputs['N-R']] = self.scale(trial['nR']) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax, self.dt,
                                                info['continue'],
                                                self.R_MISS, reward)

        if new_trial:
            self.t = 0
            self.num_tr += 1
            # compute perf
            self.perf, self.num_tr, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward, self.num_tr,
                                       self.p_stp, self.num_tr_perf, tr_perf)
            self.trial = self._new_trial(self.rng, self.dt)
        else:
            self.t += self.dt

        done = False  # TODO: revisit
        return obs, reward, done, info

        done = False  # TODO: revisit
        return obs, reward, done, info

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.95