import numpy as np
import math
import gym
import neurogym as ngym
from neurogym import spaces
from neurogym.wrappers.block import ScheduleEnvs
from neurogym.utils import scheduler
from neurogym.core import TrialWrapper
from neurogym.utils.scheduler import RandomSchedule
from neurogym.utils.scheduler import SequentialSchedule

def _get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))


def _gaussianbump(loc, theta, strength):
    dist = _get_dist(loc - theta)  # periodic boundary
    dist /= np.pi / 8
    return 0.8 * np.exp(-dist ** 2 / 2) * strength


def _cosinebump(loc, theta, strength):
    return np.cos(theta - loc) * strength / 2 + 0.5

class _MultiModalityStimulus(TrialWrapper):
    """Move observation to specific modality."""
    def __init__(self, env, modality=0, n_modality=1):
        super().__init__(env)
        self.modality = modality
        if 'stimulus' not in self.task.observation_space.name:
            raise KeyError('observation_space does not have name stimulus')
        ind_stimulus = np.array(self.task.observation_space.name['stimulus'])
        len_stimulus = len(ind_stimulus)
        ob_space = self.task.observation_space
        ob_shape = ob_space.shape[0] + (n_modality - 1) * len_stimulus
        # Shift stimulus
        arr = ind_stimulus + len_stimulus * modality
        name = {'fixation': 0,
                'stimulus': ind_stimulus + len_stimulus * modality,
                'rule': arr[-1]+1}
        #val = name['stimulus']
        #name['rule'] = val+1
        
        self.observation_space = self.task.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(ob_shape,), dtype=ob_space.dtype, name=name)
        
    def new_trial(self, **kwargs):
        return self.env.new_trial(**kwargs)

class EnvWithAdditions(ngym.TrialEnv):
          
    def _init_gt(self):
        """Initialize trial with ground_truth."""
        tmax_ind = int(self._tmax / self.dt)
        self.gt = np.zeros([tmax_ind] + list(self.action_space.shape),
                           dtype=self.action_space.dtype)
        self._gt_built = True

    def my_set_groundtruth(self, value, period=None, where=None, seq=0):
        """Set groundtruth value."""
        if not self._gt_built:
            self._init_gt()

        if where is not None:
            # TODO: Only works for Discrete action_space, make it work for Box
            value = self.action_space.name[where][value]
        if isinstance(period, str):
            for t in range(int(self.timing['decision'] / self.dt)):
                self.gt[self.start_ind['decision'] + t] = np.mod(value +int(seq* t * self.omega * self.dt), self.dim_ring)

        elif period is None:
            self.gt[:] = value
        else:
            for p in period:
                self.my_set_groundtruth(value, p)
                

class _AddFamily(EnvWithAdditions):
    
    def __init__(self,
                 dt=100,
                 rewards=None,
                 timing=None,
                 sigma=1.,
                 dim_ring=16,
                 w_mod = (1,1),
                 stim_mod = (True, True), 
                 # currently, stim_mod is tuple, turns into dict for arbitrary num of mods
                 delay_add = False,
                 delay_random = False,
                 delay = 500,
                 seq = 0):
        
        super().__init__(dt=dt)
        
        # self.w_mod1, self.w_mod2 = w_mod
        # self.stim_mod1, self.stim_mod2 = stim_mod
        self.mods = {'stim_mod' + str(i+1) : stim_mod[i] for i in range(len(stim_mod))}
        
        # access these later via self.mods[stimulus_mod1] <-- boolean 
        self.seq = seq
        self.dt = dt
        self.sigma = sigma / np.sqrt(self.dt) # Input noise
        self.dim_ring = dim_ring


        self.delay_add = delay_add
        self.delay_random = delay_random
        self.delay = delay
         
        # Rewards 
        self.rewards = {'abort':-0.1, 'correct':+1., 'fail':0.}
        if rewards: 
            self.rewards.update(rewards) 
        
        if self.delay_add: 
            self.timing = {
             'fixation': 500
            }
            for i in range(len(self.mods.keys())):
                self.timing['stim'+str(i+1)] = 500 
                if i == len(self.mods.keys())-1:  
                    pass # no delay period before decision, can be put in later
                else:
                    if self.delay_random: # const/variable/det-c/random delays? 
                        self.timing['delay'+str(i+1)] = np.random.uniform(self.delay-200, self.delay+200) 
                    else: 
                        self.timing['delay'+str(i+1)] = self.delay
            self.timing['decision'] = 200 
            
        else:
            self.timing = {
             'fixation': 500,
             'stimulus': 500,
             'decision': 200
            }
        if timing:
            self.timing.update(timing)
            
        self.abort = False
        
        self.theta = np.linspace(0, 2*np.pi, dim_ring+1)[:-1] # ignore last, since = 0 (periodic)
        self.choices = np.arange(dim_ring)
    # Spaces 
        
        
        # Observation space
        # name = {
        #     'fixation': 0,
        #     'stimulus_mod1': range(1, dim_ring+1),
        #     'stimulus_mod2': range(dim_ring+1, 2*dim_ring+1)
        # }
        name = {'fixation' : 0}
        for i in range(len(self.mods.keys())):
            name['stimulus_mod'+str(i+1)] = range(i*dim_ring+1, (i+1)*dim_ring+1)
        
        k = len(self.mods) 

        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape = (1 + k * dim_ring, ),
            dtype=np.float32,
            name=name
        )

        # Action space
        name = {
            'fixation': 0,
            'choice': range(1, dim_ring+1)
        }
        self.action_space = spaces.Discrete(1 + dim_ring, name=name)
    
    def _add_singlemod(self, trial, mod=1):
        if self.delay_add:
            stim = _gaussianbump(trial['theta' + str(mod)], self.theta, 1)
            period = 'stim' + str(mod)
            mod = '_mod' + str(mod)
            self.add_ob(stim, period=period, where='stimulus' + mod)
            
        else:
            period = 'stimulus'
            stim = _gaussianbump(trial['theta'+str(mod)], self.theta, 1) 
            mod = '_mod' + str(mod)
            self.add_ob(stim, period=period, where='stimulus'+mod)
        
        
    def _new_trial(self, **kwargs):
        #trial = {}
        #i_theta1 = self.rng.choice(self.choices) # index of theta1
        
        # while True: # gets a different index for theta2
        #     i_theta2 = self.rng.choice(self.choices) 
        #     if i_theta2 != i_theta1: 
        #         break
        
        i_thetas = {'i_theta'+str(i+1) :  self.rng.choice(self.choices) for i in range(len(self.mods))}
        trial = {'theta'+str(i+1) : self.theta[i_thetas['i_theta'+str(i+1)]] for i in range(len(self.mods))}
        # Periods 
        if self.delay_add:
            periods = ['fixation'] + list(self.timing.keys())[1:-1] + ['decision']
        else:
            periods = ['fixation', 'stimulus', 'decision']
        
        self.add_period(periods)
        
        # Observations
        self.add_ob(1, where='fixation')
        self.set_ob(0, 'decision') 
        
        if self.delay_add:
            #self.add_randn(0, self.sigma, ['stim1', 'stim2']) # add noise
            self.add_randn(0, self.sigma, ['stim'+str(i+1) for i in range(len(self.mods))])
        else:
            self.add_randn(0, self.sigma, ['stimulus']) # add noise
        # if self.stim_mod1:
        #     self._add_singlemod(trial, mod=1)
        # if self.stim_mod2:
        #     self._add_singlemod(trial, mod=2)

        for i in range(len(self.mods)): 
            if self.mods['stim_mod'+str(i+1)]: 
                num = i+1
                self._add_singlemod(trial, mod=num)
        
        
        # i_target = int(np.mod(i_theta1 + i_theta2, self.dim_ring))
        i_target = int(np.mod(sum(i_thetas.values()), self.dim_ring))

        self.set_groundtruth(i_target, period='decision', where='choice')
        
        return trial
        
    def _step(self, action): 
        new_trial = False
        gt = self.gt_now
        ob = self.ob_now
        
        # rewards
        reward = 0 
        if self.in_period('fixation'):
            if action != 0: 
                new_trial = self.abort
                reward = self.rewards['abort'] 
                
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}
    

def _add_kwargs():
    env_kwargs = {}
    return env_kwargs


def add2(**kwargs):
    env_kwargs = _add_kwargs() 
    env_kwargs.update({'stim_mod': (True, True)})
    env_kwargs.update(kwargs) 
    return _AddFamily(**env_kwargs)

def add3(**kwargs):
    env_kwargs = _add_kwargs() 
    env_kwargs.update({'stim_mod': (True, True, True)})
    env_kwargs.update(kwargs) 
    return _AddFamily(**env_kwargs)

def add4(**kwargs):
    env_kwargs = _add_kwargs() 
    env_kwargs.update({'stim_mod': (True, True, True, True)})
    env_kwargs.update(kwargs) 
    return _AddFamily(**env_kwargs) 

def add5(**kwargs):
    env_kwargs = _add_kwargs() 
    env_kwargs.update({'stim_mod': (True, True, True, True, True)})
    env_kwargs.update(kwargs) 
    return _AddFamily(**env_kwargs)

def _dlyadd_kwargs(): 
    env_kwargs = {'delay_add': True, 'delay': 500}
    return env_kwargs

def dlyadd2(**kwargs):
    env_kwargs = _dlyadd_kwargs() 
    env_kwargs.update({'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    return _AddFamily(**env_kwargs)

def dlyadd3(**kwargs):
    env_kwargs = _dlyadd_kwargs() 
    env_kwargs.update({'stim_mod': (True, True, True)})
    env_kwargs.update(kwargs)
    return _AddFamily(**env_kwargs)

def dlyadd4(**kwargs):
    env_kwargs = _dlyadd_kwargs() 
    env_kwargs.update({'stim_mod': (True, True, True, True)})
    env_kwargs.update(kwargs)
    return _AddFamily(**env_kwargs)

def dlyadd5(**kwargs):
    env_kwargs = _dlyadd_kwargs() 
    env_kwargs.update({'stim_mod': (True, True, True, True, True)})
    env_kwargs.update(kwargs)
    return _AddFamily(**env_kwargs)

def rdlyadd2(**kwargs):
    env_kwargs = _dlyadd_kwargs() 
    env_kwargs.update({'delay_random': True, 'stim_mod': (True, True)})
    env_kwargs.update(kwargs)
    return _AddFamily(**env_kwargs)

def rdlyadd3(**kwargs):
    env_kwargs = _dlyadd_kwargs() 
    env_kwargs.update({'delay_random': True, 'stim_mod': (True, True, True)})
    env_kwargs.update(kwargs)
    return _AddFamily(**env_kwargs)

def rdlyadd4(**kwargs):
    env_kwargs = _dlyadd_kwargs() 
    env_kwargs.update({'delay_random': True, 'stim_mod': (True, True, True, True)})
    env_kwargs.update(kwargs)
    return _AddFamily(**env_kwargs)

def rdlyadd5(**kwargs):
    env_kwargs = _dlyadd_kwargs() 
    env_kwargs.update({'delay_random': True, 'stim_mod': (True, True, True, True, True)})
    env_kwargs.update(kwargs)
    return _AddFamily(**env_kwargs)


    
        