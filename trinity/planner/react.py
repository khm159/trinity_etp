import os

class React():
    def __init__(self, cfg, llm_handler):
        self.cfg = cfg
        self.llm_handler = llm_handler
        self.max_steps = cfg.agent.max_steps
        self.max_decisions = cfg.agent.max_decisions
        self.cur_step_id = 1
        self.cur_decision_id = 1
        self._init(cfg)
    
    def _init(self, cfg):
        raise NotImplementedError()

    def run(self, task_d, log):
        raise NotImplementedError()
    
    def collect(self, task_d):
        raise NotImplementedError()