import os
import json
from .react import React

class TrinityPreact(React):
    def _init(self, cfg):
        """overriding and modified initialization in  class"""
        self.text_trajectory_save_dir = os.path.join(
            cfg.log.save_path,
            "text_trajectory"
        )
        if not os.path.exists(self.text_trajectory_save_dir):
            os.makedirs(self.text_trajectory_save_dir)
        self.reflection_prompt = None

    def run(self, env, scene_data, log):
        # load ThorEnv
        collect_file_path = os.path.join(
            self.text_trajectory_save_dir, 
            scene_data['sample_name']
        )
        # Generate log data dictionary 
        log_data = dict()
        log_data['trajectory'] = "" 
        log_data['example'] = ""
        log_data['prompt'] = ""

        # Get instruction and human attribute strings  
        instruction_str = f'Your task is to: {env.instruction}' 
        if "human_attributes" in scene_data.keys():
            human_attributes_str = scene_data['human_attributes']

        # Logging informations 

        # - 1. save instruction 
        log_data['instruction'] = instruction_str
        trajectory = f'{instruction_str}\n'

        # - 2. save human attributes information if exists
        if "human_attributes" in scene_data.keys():
            log_data['human_attributes'] = scene_data['human_attributes']
            trajectory += f'{human_attributes_str}\n'

        # initial react observation logging 
        init_obs = env.reset(scene_data)

        if "human_attributes" not in scene_data.keys():
            nl_inst_info = {
                'nl_inst': instruction_str, 
                'task_type': scene_data['task_type'],
                'message': None,
                'human_attributes': None
            }
        else:
            nl_inst_info = {
                'nl_inst': instruction_str, 
                'task_type': scene_data['task_type'],
                'message': None,
                'human_attributes': human_attributes_str
            }

        # Reset LLM Handler --> set user prompt 
        if self.reflection_prompt is None:
            # if the reflexion is not done yet, apply initial prompt
            self.llm_handler.reset(nl_inst_info, init_obs)
        else:
            print("Aplly Reflexion prompt!")
            NotImplementedError

        # - 3. logging initial prompt and example
        log_data['example'] = self.llm_handler.ic_ex_prompt
        log_data['prompt'] = self.llm_handler.prompt

        # Get initial skill set from initial observation information 
        skill_set = self.llm_handler.update_skill_set(init_obs)
        init_obs_str = init_obs['text']
        
        # logging instruction to logger
        log.info(f'{instruction_str}')
        if "human_attributes" in scene_data.keys():
            log.info(f'{human_attributes_str}')

        # - 4. update initial observation to trajectory 
        trajectory +=f'{init_obs_str}\n'

        # start evaluation the given task 
        self.cur_step_id, self.cur_decision_id = 1, 1
        steps =[]
        
        while True:
            if self.cur_step_id > self.max_steps:
                # - 5. logging Max Step Errors 
                log.info('Max steps')
                trajectory += f'Max steps\n'
                log_data['trajectory'] = trajectory # save total trajectory
                terminate_info = {
                    'terminate': 'max_step', 
                    'step_id': self.cur_step_id, 
                    'decision_id': self.cur_decision_id,
                    'init_obs': init_obs,
                    'nl_inst_info': nl_inst_info,
                    'trajectory': trajectory,
                    'collect_file_path': collect_file_path
                }
                # save log data to json file 
                with open(collect_file_path, 'w') as f:
                    # dump 
                    json.dump(log_data, f, indent=4)
                return terminate_info
            
            if self.cur_decision_id > self.max_decisions:
                # - 5. logging Max Decision Errors (think too much)
                log.info('Max decisions')
                trajectory += f'Max decisions\n'
                log_data['trajectory'] = trajectory # save total trajectory
                terminate_info = {
                    'terminate': 'max_decision', 
                    'step_id': self.cur_step_id, 
                    'decision_id': self.cur_decision_id,
                    'init_obs': init_obs,
                    'nl_inst_info': nl_inst_info,
                    'trajectory': trajectory,
                    'collect_file_path': collect_file_path
                }
                # save log data to json file 
                with open(collect_file_path, 'w') as f:
                    # dump 
                    json.dump(log_data, f, indent=4)
                return terminate_info
            ################## Next Step Planning Stage 
            try:
                next_step_info = self.llm_handler.plan_next_step(skill_set, steps)

                action_step = next_step_info['next_step']
                thought = next_step_info['thought']
                next_step_class = next_step_info['next_step_class']
                prediction = next_step_info['prediction']

                if action_step is None:
                    # fail to generate proper json 
                    # treat as think case 
                    log.info(f'Plan Next Step Error. Maybe the output is not propse JSON structure. Treat as Think step: {action_step}')
                    log.info(f'Think: {thought}')
                    trajectory += f'Think: {thought}\n'
                else:
                    steps.append(next_step_class)
                    log.info(f'Think: {thought}') 
                    log.info(f'{next_step_class}: {action_step}')
                    log.info(f'Predict: {prediction}')

                    # - 5. logging inferenced next step information
                    trajectory += f'Think: {thought}\n'
                    trajectory += f'{next_step_class}: {action_step}\n'
                    trajectory += f'Predict: {prediction}\n'

            except Exception as error_message:
                ### TODO: except case
                trajectory += f'Plan Next Step Error: {error_message}\n'
                log_data['trajectory'] = trajectory # save total trajectory
                log.info(f"Plan Next Step Error: {error_message}")
                terminate_info = {
                    'terminate': 'plan_next_step_error', 
                    'step_id': self.cur_step_id, 
                    'decision_id': self.cur_decision_id,
                    'trajectory': trajectory,
                    'init_obs': init_obs,
                    'nl_inst_info': nl_inst_info
                }
                # save log data to json file 
                with open(collect_file_path, 'w') as f:
                    # dump 
                    json.dump(log_data, f, indent=4)
                return terminate_info
            
            ################## Next Step Execution Stage
            if next_step_class == 'Think':
                self.cur_decision_id += 1
                pass

            elif next_step_class == 'Act':
                prediction = next_step_info['prediction']
                if action_step == 'done':
                    # - 5. logging Done Step
                    trajectory += f'{action_step}\n'
                    log_data['trajectory'] = trajectory # save total trajectory
                    log.info('Done')
                    terminate_info = {
                        'terminate': 'done', 
                        'step_id': self.cur_step_id, 
                        'decision_id': self.cur_decision_id,
                        'trajectory': trajectory,
                        'init_obs': init_obs,
                        'nl_inst_info': nl_inst_info,
                        'collect_file_path': collect_file_path
                    }
                    # save log data to json file 
                    with open(collect_file_path, 'w') as f:
                        # dump 
                        json.dump(log_data, f, indent=4)
                    
                    return terminate_info
                
                elif action_step == 'failure':
                    trajectory += f'{action_step}\n'
                    log_data['trajectory'] = trajectory # save total trajectory
                    log.info('Failure')
                    terminate_info = {
                        'terminate': 'failure', 
                        'step_id': self.cur_step_id, 
                        'decision_id': self.cur_decision_id,
                        'trajectory': trajectory,
                        'init_obs': init_obs,
                        'nl_inst_info': nl_inst_info,
                        'collect_file_path': collect_file_path
                    }
                    # save log data to json file 
                    with open(collect_file_path, 'w') as f:
                        # dump 
                        json.dump(log_data, f, indent=4)
                    
                    return terminate_info
                else:
                    obs = env.llm_skill_interact(action_step)
                    self.llm_handler.add_obs(obs['message'])
                    self.cur_step_id += 1
                    self.cur_decision_id +=1

                    # - 5. logging action result
                    trajectory += f'{obs["message"]}\n'
                    log.info(obs['message'])
                    
                    # Update skill set using partial observation 
                    if self.cfg.trinity.is_hae_dataset:
                        skill_set = self.llm_handler.update_skill_set(obs, True)
                    else:
                        skill_set = self.llm_handler.update_skill_set(obs)

            elif next_step_class == 'Error':
                self.cur_step_id += 1
                self.cur_decision_id +=1
                # import pdb; pdb.set_trace()
            else:
                raise NotImplementedError()
        