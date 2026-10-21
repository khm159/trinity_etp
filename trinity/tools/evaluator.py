import os
import json
import sys
import copy
import datetime
import pickle
import logging
import time

sys.path.insert(0, '..')
sys.path.insert(0, '')


from trinity.llm.trinity_llm_handler import TrinityLLMHandler, TrinityActOnlyLLMHandler, TrinityPreActLLMHandler, TrinityStateActLLMHandler
# ---------------- Act Agent -------------------------
from trinity.planner.trinity_act import TrinityAct
# ---------------- PreAct Agent -------------------------
from trinity.planner.trinity_preact import TrinityPreact
# ---------------- StateAct Agent -------------------------
from trinity.planner.trinity_stateact import TrinityStateAct
from trinity.env.thor_env import ThorEnv5

from omegaconf import OmegaConf

log = logging.getLogger(__name__)
class Evaluator():
    def __init__(self, cfg):
        self.cfg = cfg
    
    def evaluate(self):
        # log config
        env=None
        log.info(OmegaConf.to_yaml(self.cfg))

        if self.cfg.agent.agent_type == 'react':
            trinity_llm_handler = TrinityLLMHandler(self.cfg)
            if self.cfg.agent.think_every_step:
                from trinity.planner.trinity_react_think_every_step import TrinityReact
                tp = TrinityReact(self.cfg, trinity_llm_handler)
            else:
                from trinity.planner.trinity_react import TrinityReact
                tp = TrinityReact(self.cfg, trinity_llm_handler)

        elif self.cfg.agent.agent_type == 'preact_p':
            if not self.cfg.agent.think_every_step:
                raise ValueError("preact_p agent must use think every step")
            else:
                trinity_llm_handler = TrinityPreActLLMHandler(self.cfg)
                tp = TrinityPreact(self.cfg, trinity_llm_handler)

        elif self.cfg.agent.agent_type == 'stateact':
            trinity_llm_handler = TrinityStateActLLMHandler(self.cfg)
            tp = TrinityStateAct(self.cfg, trinity_llm_handler)
            
        elif self.cfg.agent.agent_type == 'act':
            trinity_llm_handler = TrinityActOnlyLLMHandler(self.cfg)
            tp = TrinityAct(self.cfg, trinity_llm_handler)

        else:
            NotImplementedError
        
        eval_dataset_path = os.path.join(
            self.cfg.trinity.dataset_path,
            self.cfg.trinity.eval_set
        )
        self.eval_dataset_list = []
        json_dir_list = os.listdir(eval_dataset_path)
        for dir in json_dir_list:
            json_list = os.listdir(
                os.path.join(eval_dataset_path, dir)
            )
            for json_name in json_list:
                json_path = os.path.join(dir, json_name)
                sample_name = dir.split("/")[-1]
                sample_name = sample_name+"_"+json_name
                self.eval_dataset_list.append(
                    {
                        'json_name':json_name,
                        'sample_name':sample_name,
                        'path':json_path
                    }
                    
                )
        results = []
        cam_frame_save_path_base = None

        for sample_info in self.eval_dataset_list:
            sample_name = sample_info['sample_name']
            is_exist = False
            scene_data = self.load_json(
                os.path.join(
                    eval_dataset_path,
                    sample_info['path']
                ) 
            )

            print("--> Scene Name : ", sample_info['sample_name'])
            log.info(f"Eval Start : {sample_name}")

            # 3. load envrinment 
            if env is None:
                # create new environment   
                scene_data['sample_name']=sample_info['sample_name']
                scene_data['json_name'] = sample_info['path'].split("/")[-1]
                env = ThorEnv5(self.cfg, scene_data)
            else:
                # reset 
                scene_data['sample_name']=sample_info['sample_name']
                scene_data['json_name'] = sample_info['path'].split("/")[-1]
                env.reset(scene_data, is_reflexion=False)
            
            # set trajectory image path 
            if cam_frame_save_path_base is None:
                cam_frame_save_path_base = copy.deepcopy(env.cam_frame_save_path)
            
            cam_frame_save_path = os.path.join(
                cam_frame_save_path_base, 
                self.cfg.name,
                sample_name
            )
            if not os.path.exists(cam_frame_save_path):
                os.makedirs(cam_frame_save_path)

            env.cam_frame_save_path = cam_frame_save_path
            terminate_info = tp.run(env, scene_data, log)

            print(terminate_info)
            print(f'Evaluation task {sample_name} is finished.')
            
            # evaluate results 
            goal_results = env.goal_checker.check_goal(env.controller.last_event)

            if isinstance(goal_results, list):
                # if all True then success
                is_success= False
                if not False in goal_results:
                    is_success=True
            elif isinstance(goal_results, tuple):
                is_success=False
                is_checked=False
                for goal_result in goal_results:
                    if not goal_result:
                        is_success=False
                        is_checked=True
                        break
                if not is_checked:
                    is_success=True
            else:
                print(type(goal_results))

            log.info(f"Eval Done : {sample_name}")
            log.info(f"Eval Result : {is_success}")
            results.append(is_success)

            # save results 
            collect_file_path = terminate_info['collect_file_path']

            # laod json and add result 
            with open(collect_file_path, 'r') as f:
                collect_data = json.load(f)
            collect_data['is_success'] = is_success
            if is_success:
                tail = "-success.json"
            else:
                tail = "-fail.json"
            with open(collect_file_path.split(".json")[0]+tail, 'w') as f:
                json.dump(
                    collect_data, 
                    f, 
                    indent=4
                )
                
        # evaluate 
        print("results", results)
        log.info(results)
        n = len(results)
        n_success = 0
        for e in results:
            if e:
                n_success += 1
        log.info(f'success rate: {n_success / n * 100:.2f} % ({n_success}/{n})')
        log.info('------------------------')
    
    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

