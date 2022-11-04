import numpy as np
from gym import Wrapper
from neurogym import spaces
import neurogym as ngym

class DelayedRuleInput(ngym.TrialWrapper):
    """
    Modifies environment by adding a new period (at the end of a trial) with the rule input, and no other inputs.
    """
    def __init__(self, env, rule_input_dur=200):
        super().__init__(env)
        self.env = env.unwrapped # key 
        
        # reconfigure ob space 
        env_oss = self.task.observation_space.shape[0]
        new_ob_dim = env_oss + 1 # add ob dim for rule input 
        
        
        name_new = self.task.observation_space.name
        
        # do not want to make any reference to specific stimuli, since these will change 
        vals = name_new.values()
        lst = list(vals)
        if np.isscalar(lst[-1]):
            name_new['rule'] = lst[-1]+1
        else:
            name_new['rule'] = lst[-1][-1]+1
        
        #print(dir(self.env))
                    
        # new observation space, expanded dim, new names
        self.observation_space = self.task.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(new_ob_dim,),
                                            dtype=np.float32,
                                            name=name_new)
        
        
        self.timing['stimulus'] += rule_input_dur # offset to compensate for length of rule input 
       # self.unwrapped.timing['fixation'] += rule_input_dur
        
        # add rule period of specified duration 
        self.add_period('rule', after=self.unwrapped.timing['fixation']+self.unwrapped.timing['stimulus']-rule_input_dur, duration=rule_input_dur)
        #self.env.unwrapped.add_period('rule', after=self.unwrapped.timing['stimulus'], duration=rule_input_dur)
        
    def new_trial(self, **kwargs): 
        # Rule observation
        self.set_ob(0, where='fixation')
        
        
        # Turn off fixation and stimulus inputs during rule 
        self.set_ob(0, period='rule', where='fixation')
        self.set_ob(0, period='rule', where='stimulus')
        
        
        self.env.unwrapped.set_ob(0, where='fixation')
        
        #print(self.env.unwrapped.unwrapped)
        
        
        return self.env.new_trial(**kwargs)

        
        
        