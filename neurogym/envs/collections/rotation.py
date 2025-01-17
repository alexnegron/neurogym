import numpy as np
import math
import gym
import matplotlib.pyplot as plt
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

class _Rotation(EnvWithAdditions):
    def __init__(self,
                 dt=100,
                 rewards=None,
                 timing=None,
                 sigma=.15,
                 dim_ring=16,
                 stim_mod = (True, True, True), 
                 # currently, stim_mod is tuple, turns into dict for arbitrary num of mods
                 seq = 0, 
                 delay = 0,
                 t = False,
                 m = False,
                 rot = (False, False, False), 
                 rot_speed = (1, 1, 1),
                 state = ('lp', 'lp', 'lp'),
                 varbump = (False, False, False),
                 rot_comp = False):
        
        super().__init__(dt=dt)

        self.mods = {'stim_mod' + str(i+1) : stim_mod[i] for i in range(len(stim_mod))}
        self.rot = {'stim_mod' + str(i+1) : rot[i] for i in range(len(rot))}
        self.rot_speed = {'stim_mod' + str(i+1) : rot_speed[i] for i in range(len(rot_speed))}
        self.state = {'stim_mod' + str(i+1) : state[i] for i in range(len(state))}
        self.varbump = {'stim_mod' + str(i+1) : varbump[i] for i in range(len(varbump))}
        self.seq = seq
        self.dt = dt
        self.sigma = sigma / np.sqrt(self.dt) # Input noise
        self.dim_ring = dim_ring
        self.delay = delay
        self.t = t
        self.m = m
        self.rot_comp = rot_comp
        self.lp = 0


        # Rewards 
        self.rewards = {'abort':-0.1, 'correct':+1., 'fail':0.}
        if rewards: 
            self.rewards.update(rewards) 
        
        # Timing
        self.timing['fixation'] = 500
        # mod_durs = [np.random.uniform(100, 800) for _ in range(1, len(self.mods))]
        mod_durs = []
        self.timing['stimulus'+str(1)] = 900
        for i in range(1, len(self.mods)): # skip first ring, set it at end
            dur = np.random.uniform(400, 800)
            mod_durs.append(dur)
            self.timing['stimulus'+str(i+1)] = dur
            if self.delay > 0: 
                if i == len(self.mods.keys())-1:
                    pass
                else:
                    self.timing['delay'+str(i)] = self.delay

        #self.timing['stimulus'+str(1)] = np.max(mod_durs)
        self.timing['decision'] = 200
        if timing:
            self.timing.update(timing)
            
        self.abort = False
        self.theta = np.linspace(0, 2*np.pi, self.dim_ring + 1)[:-1] # ignore last, since 2pi = 0 (periodic)
        self.choices = np.arange(self.dim_ring) 
        
    # Spaces 
        
        # Observation space
        name = {'fixation' : 0}
        
        # stimulus_mod1 is the ring the other mods will be rotating
        for i in range(len(self.mods)): 
            name['stimulus_mod'+str(i+1)] = range(i*dim_ring + 1, (i+1)*dim_ring + 1)
        
        num_mods = len(self.mods)
        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape = (1 + num_mods*dim_ring, ),
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
        period = 'stimulus'+str(mod)
        #period = 'stimulus1'
        stim = _gaussianbump(trial['theta' + str(mod)], self.theta, 1) 
        mod = '_mod' + str(mod) 
        self.add_ob(stim, period=period, where='stimulus' + mod)

        
    def _new_trial(self, **kwargs):
        i_thetas = {'i_theta'+str(i+1) :  self.rng.choice(self.choices) for i in range(len(self.mods))} # Place bumps randomly on each ring
        # Note: mod1 initial position is random; should this be fixed?  
        trial = {'theta'+str(i+1) : self.theta[i_thetas['i_theta'+str(i+1)]] for i in range(len(self.mods))}
        
        # Timing
        # note: want each trial to have different timings of rotator ring presentations 
        mod_durs = []
        for i in range(1, len(self.mods)):
            dur = np.random.uniform(500, 1200)
            mod_durs.append(dur)
            self.timing['stimulus'+str(i+1)] = dur
            if self.delay > 0: 
                if i == len(self.mods.keys())-1:
                    pass
                else:
                    self.timing['delay'+str(i)] = self.delay

        # Periods 
        periods = list(self.timing.keys())
        self.add_period(periods)
        
        # stimulus_mod1 on for the full duration
        stim = _gaussianbump(trial['theta1'], self.theta, 1) 
        self.add_ob(stim, where='stimulus_mod1')

        # add rotator rings 
        for i in range(1, len(self.mods)): 
            if self.mods['stim_mod'+str(i+1)]: 
                num = i+1
                self._add_singlemod(trial, mod=num)
        
        # perform task-prescribed modifications to each ring 
        rot_dirs = {} # keep track of which directions ring rotates
        for i in range(len(self.mods)):
            if self.rot['stim_mod'+str(i+1)]:
                dir = np.random.choice([-1,1])
                rot_dirs['stim_mod'+str(i+1)] = dir

            else: # when ring is NOT rotating, direction it rotates ring 1 by depends where theta falls on the ring
                if 0 <=  trial['theta'+str(i+1)] and trial['theta'+str(i+1)] < np.pi: # upper half-circle => counter-clockwise rotate
                    dir = 1
                else: # lower half-circle => clockwise rotation
                    dir = -1
                rot_dirs['stim_mod'+str(i+1)] = dir

            speed = self.rot_speed['stim_mod'+str(i+1)]
            obs = self.ob[self.start_ind['stimulus1'] : self.end_ind['stimulus'+str(len(self.mods))]]
            lp_idx = i_thetas['i_theta'+str(i+1)] # bump idx tracker (used for lp tasks)
            dir = rot_dirs['stim_mod'+str(i+1)]

            for j in range(obs.shape[0]):
                if self.rot['stim_mod'+str(i+1)]: # mod is rotating
                    obs[j, i*self.dim_ring+1 : (i+1)*self.dim_ring+1] = np.roll(obs[j, i*self.dim_ring+1 : (i+1)*self.dim_ring+1], dir*speed*j) # performs rotation
                    lp_idx = np.mod(lp_idx + dir*speed*1, self.dim_ring) # track the index of bump as rotation occurs 
                
                if self.varbump['stim_mod'+str(i+1)]: # variable bump magnitude 
                    x = np.random.uniform(1, 2)
                    obs[j, i*self.dim_ring+1 : (i+1)*self.dim_ring+1] = x*obs[j, i*self.dim_ring+1 : (i+1)*self.dim_ring+1] # stretch gaussian by x

                # modify ring states: set i_theta according to task demands
                if self.state['stim_mod'+str(i+1)] == 'fp': 
                    pass # ring state at initialization is what we want, don't need to change
                elif self.state['stim_mod'+str(i+1)] == 'lp':
                    i_thetas['i_theta'+str(i+1)] = lp_idx 
                elif self.state['stim_mod'+str(i+1)] == 'min':
                    i_thetas['i_theta'+str(i+1)] = np.argmin(obs[j, i*self.dim_ring+1 : (i+1)*self.dim_ring+1]) 
                elif self.state['stim_mod'+str(i+1)] == 'max':
                    i_thetas['i_theta'+str(i+1)] = np.argmax(obs[j, i*self.dim_ring+1 : (i+1)*self.dim_ring+1]) 
                else: 
                    pass 
        
        self.add_ob(1, where='fixation')
        self.set_ob(0, 'decision')
        self.set_ob(0, period='fixation', where='stimulus_mod1')
        
        # add noise 
        self.add_randn(0, self.sigma, ['stimulus'+str(i+1) for i in range(len(self.mods))])
        if self.delay > 0: 
            self.add_randn(0, self.sigma, ['delay'+str(i) for i in range(1, len(self.mods) - 1)])
        
        # Compute how much each mod should rotate mod1 
        sum = 0
        for i in range(1, len(self.mods)):
            r = 1
            m = 1
            comp = 1 

            if self.t: 
                r = math.floor((self.timing['stimulus'+str(i+1)]+50)/self.dt) # magnitude is proportional to timing 

            if self.m: # case where ring state factors into magnitude of rotation 
                if self.rot['stim_mod'+str(i+1)]: 
                    m = i_thetas['i_theta'+str(i+1)]

                else: # in non-rotating case, direction of rotation depends on state being upper/lower half ring
                    if rot_dirs['stim_mod'+str(i+1)] > 0: # counter-clockwise 
                        m = i_thetas['i_theta'+str(i+1)] 
                    else:  # clockwise, so m should be wrt "opening in reverse direction"
                        m = self.dim_ring - i_thetas['i_theta'+str(i+1)] 

            if self.rot_comp: # comparing rotation speeds; only the fastest-rotating ring rotates mod1 
                comp = int(self.rot_speed['stim_mod' + str(i+1)] == max(list(self.rot_speed.values())))
            

            sum += rot_dirs['stim_mod'+str(i+1)] * m * r * comp
        
        i_target = np.mod(i_thetas['i_theta1'] + sum, self.dim_ring)
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


class _ContextRotation(EnvWithAdditions):
    def __init__(self,
                 dt=100,
                 rewards=None,
                 timing=None,
                 sigma=.15,
                 dim_ring=16,
                 stim_mod = (True, True), 
                 # currently, stim_mod is tuple, turns into dict for arbitrary num of mods
                 delay = 0,
                 t = False,
                 m = False,
                 rot = (False, False), 
                 rot_speed = (1, 1),
                 state = ('fp', 'fp'),
                 varbump = (False, False),
                 ctx = 1,
                 rot_comp = False):
        
        super().__init__(dt=dt)

        self.mods = {'stim_mod' + str(i+1) : stim_mod[i] for i in range(len(stim_mod))}
        self.rot = {'stim_mod' + str(i+1) : rot[i] for i in range(len(rot))}
        self.rot_speed = {'stim_mod' + str(i+1) : rot_speed[i] for i in range(len(rot_speed))}
        self.state = {'stim_mod' + str(i+1) : state[i] for i in range(len(state))}
        self.varbump = {'stim_mod' + str(i+1) : varbump[i] for i in range(len(varbump))}
        self.dt = dt
        self.sigma = sigma / np.sqrt(self.dt) # Input noise
        self.dim_ring = dim_ring
        self.delay = delay
        self.t = t

        self.m = m
        self.rot_comp = rot_comp
        self.lp = 0
        self.ctx = ctx


        # Rewards 
        self.rewards = {'abort':-0.1, 'correct':+1., 'fail':0.}
        if rewards: 
            self.rewards.update(rewards) 
        
        # Timing
        self.timing['fixation'] = 500
        # mod_durs = [np.random.uniform(100, 800) for _ in range(1, len(self.mods))]
        mod_durs = []
        #self.timing['stimulus'+str(1)] = 900
        dur = 500
        for i in range(len(self.mods)): # skip first ring, set it at end
            #dur = np.random.uniform(400, 800)
            mod_durs.append(dur)
            self.timing['stimulus'+str(i+1)] = dur
            if self.delay > 0: 
                if i == len(self.mods.keys())-1:
                    pass
                else:
                    self.timing['delay'+str(i)] = self.delay

        #self.timing['stimulus'+str(1)] = np.max(mod_durs)
        self.timing['context'] = 200
        self.timing['decision'] = 200
        if timing:
            self.timing.update(timing)
            
        self.abort = False
        self.theta = np.linspace(0, 2*np.pi, self.dim_ring + 1)[:-1] # ignore last, since 2pi = 0 (periodic)
        self.choices = np.arange(self.dim_ring) 
        
    # Spaces 
        
        # Observation space
        name = {'fixation' : 0}
        
        # Set up names for each of the stimulus mods
        for i in range(len(self.mods)): 
            name['stimulus_mod'+str(i+1)] = range(i*dim_ring + 1, (i+1)*dim_ring + 1)
        
        # Set up name for contextual input
        last = name['stimulus_mod'+str(len(self.mods))][-1]
        for i in range(len(self.mods)):
            name['context'+str(i+1)] = 1+last+i
        
        num_mods = len(self.mods)
        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape = (1 + num_mods*dim_ring + len(self.mods), ), #  + len(self.mods) for the context cues
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
        period = 'stimulus'+str(mod)
        stim = _gaussianbump(trial['theta' + str(mod)], self.theta, 1) 
        mod = '_mod' + str(mod) 
        self.add_ob(stim, period=period, where='stimulus' + mod)

        
    def _new_trial(self, **kwargs):
        i_thetas = {'i_theta'+str(i+1) :  self.rng.choice(self.choices) for i in range(len(self.mods))} # Place bumps randomly on each ring
        
        # Note: mod1 initial position is random; should this be fixed?  
        trial = {'theta'+str(i+1) : self.theta[i_thetas['i_theta'+str(i+1)]] for i in range(len(self.mods))}
        
        # Timing
        # note: want each trial to have different timings of rotator ring presentations 
        mod_durs = []
        #dur = np.random.uniform(500, 1200)
        dur = 500
        if self.delay > 0: 
            for i in range(len(self.mods)):
                mod_durs.append(dur)
                self.timing['stimulus'+str(i+1)] = dur
                if self.delay > 0: 
                    if i == len(self.mods.keys())-1:
                        pass
                    else:
                        self.timing['delay'+str(i)] = self.delay
        else:
            self.timing['stimulus'] = dur
            
        # Periods 
        periods = list(self.timing.keys())
        self.add_period(periods)

        # Add rotating rings 
        for i in range(len(self.mods)): 
            if self.mods['stim_mod'+str(i+1)]: 
                num = i+1
                self._add_singlemod(trial, mod=num)
        
        # perform task-prescribed modifications to each ring 
        rot_dirs = {} # keep track of which directions ring rotates
        for i in range(len(self.mods)):
            dir = np.random.choice([-1,1])
            #dir = 1 
            rot_dirs['stim_mod'+str(i+1)] = dir
            speed = self.rot_speed['stim_mod'+str(i+1)]
            obs = self.ob[self.start_ind['stimulus'+str(i+1)] : self.end_ind['stimulus'+str(i+1)]]
            lp_idx = i_thetas['i_theta'+str(i+1)] # bump idx tracker (used for lp tasks)
            dir = rot_dirs['stim_mod'+str(i+1)]

            
            for j in range(obs.shape[0]):
                if self.rot['stim_mod'+str(i+1)]: # mod is rotating
                    obs[j, i*self.dim_ring+1 : (i+1)*self.dim_ring+1] = np.roll(obs[j, i*self.dim_ring+1 : (i+1)*self.dim_ring+1], dir*speed*j) # performs rotation
            
            prev = i_thetas['i_theta'+str(i+1)]
            if dir == 1: 
                correction = -1
            else:
                correction = 1 
            i_thetas['i_theta'+str(i+1)] = np.mod(prev + dir*speed*dur//self.dt + speed*correction, self.dim_ring)

        self.add_ob(1, where='fixation')
        self.set_ob(0, 'decision')
        self.set_ob(0, period='fixation', where='stimulus_mod1')

        # add noise 
        self.add_randn(0, self.sigma, ['stimulus'+str(i+1) for i in range(len(self.mods))])
        if self.delay > 0: 
            self.add_randn(0, self.sigma, ['delay'+str(i) for i in range(1, len(self.mods) - 1)])
        
        #ctx_idx = np.random.choice(1,range(len(self.mods)+1))
        self.set_ob(0, period='context')
        self.set_ob(1, period='context', where='context'+str(self.ctx))

        for i in range(len(self.mods)):
            for j in range(len(self.mods)):
                self.set_ob(0, period='stimulus'+str(i+1), where='context'+str(j+1))


        # self.pos = i_thetas['i_theta'+str(ctx_idx+1)]
        # self.ctx = ctx_idx 
        i_target = i_thetas['i_theta'+str(self.ctx)] # target is determined by context
        self.i_target = i_target
        
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


# Task definitions 
def _rot_kwargs():
    env_kwargs = {}
    return env_kwargs

def _trot_kwargs():
    env_kwargs = {'t' : True}
    return env_kwargs

def _mrot_kwargs():
    env_kwargs = {'m' : True}
    return env_kwargs

def _mtrot_kwargs():
    env_kwargs = {'m' : True, 't' : True}
    return env_kwargs

def _dlytrot_kwargs():
    env_kwargs = {'delay' : 300} # default delay length is 300ms 
    return env_kwargs

def _2ring_ctxrot_kwargs():
    env_kwargs = {'stim_mod':(True, True),'rot':(True, True), 'state':('lp','lp'), 'sigma':0}
    return env_kwargs

def _3ring_ctxrot_kwargs():
    env_kwargs = {'stim_mod':(True, True, True),
                  'rot':(True, True, True), 
                  'state':('lp','lp', 'lp'), 
                  'sigma':0, 
                  'rot_speed':(1,1,1)}
    return env_kwargs

"""
Task names
"""
##########
def dlyctxrot1_2ring(**kwargs):
    env_kwargs = _2ring_ctxrot_kwargs()
    env_kwargs.update({'delay':500, 'ctx':1})
    env_kwargs.update(**kwargs)
    return _ContextRotation(**env_kwargs)

def dlyctxrot2_2ring(**kwargs):
    env_kwargs = _2ring_ctxrot_kwargs()
    env_kwargs.update({'delay':500, 'ctx':2})
    env_kwargs.update(**kwargs)
    return _ContextRotation(**env_kwargs)

def dlyctxrot1_3ring(**kwargs):
    env_kwargs = _3ring_ctxrot_kwargs()
    env_kwargs.update({'delay':500, 'ctx':1})
    env_kwargs.update(**kwargs)
    return _ContextRotation(**env_kwargs)

def dlyctxrot2_3ring(**kwargs):
    env_kwargs = _3ring_ctxrot_kwargs()
    env_kwargs.update({'delay':500, 'ctx':2})
    env_kwargs.update(**kwargs)
    return _ContextRotation(**env_kwargs)

def dlyctxrot3_3ring(**kwargs):
    env_kwargs = _3ring_ctxrot_kwargs()
    env_kwargs.update({'delay':500, 'ctx':3})
    env_kwargs.update(**kwargs)
    return _ContextRotation(**env_kwargs)

#### 



