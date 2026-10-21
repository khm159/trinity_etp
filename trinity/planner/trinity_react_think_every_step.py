import os
import json
from .react import React

class TrinityReact(React):
    def _init(self, cfg):
        """overriding and modified initialization in React class"""
        self.text_trajectory_save_dir = os.path.join(
            cfg.log.save_path,
            "text_trajectory"
        )
        if not os.path.exists(self.text_trajectory_save_dir):
            os.makedirs(self.text_trajectory_save_dir)
        self.reflection_prompt = None

    def run(self, env, scene_data, log, is_reflexion=False):
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

                if action_step is None:
                    # fail to generate proper json 
                    # treat as think case 
                    log.info(f'Plan Next Step Error. Maybe the output is not propse JSON structure. Treat as Think step: {action_step}')
                    log.info(f'Think: {thought}')
                    trajectory += f'Think: {thought}\nOK.\n'
                else:
                    steps.append(next_step_class)
                    log.info(f'Think: {thought}') 
                    log.info(f'{next_step_class}: {action_step}')

                    # - 5. logging inferenced next step information
                    trajectory += f'Think: {thought}\nOK.\n'
                    trajectory += f'{next_step_class}: {action_step}\n'

            except Exception as error_message:
                ### TODO: except case
                trajectory += f'Plan Next Step Error: {error_message}\n'
                log_data['trajectory'] = trajectory # save total trajectory
                terminate_info = {
                    'terminate': 'plan_next_step_error', 
                    'step_id': self.cur_step_id, 
                    'decision_id': self.cur_decision_id,
                    'trajectory': trajectory,
                    'init_obs': init_obs,
                    'nl_inst_info': nl_inst_info,
                    'collect_file_path': collect_file_path
                }
                log.info(f"Plan Next Step Error: {error_message}")
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
                if action_step == 'done':
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
            else:
                raise NotImplementedError()
        
    def collect(self, env, scene_data,is_reflexion=False):
        # set text trajectory save directory
        self.text_trajectory_save_dir = self.cfg.collection.save_path
        if not os.path.exists(self.text_trajectory_save_dir):
            os.makedirs(self.text_trajectory_save_dir)

        # TODO: implement this
        avail_skill_set = [
            "go to", "pick up", "put down", "open", "close", 
            "turn on", "turn off", "slice", "drop", "throw",
            "break", "pour", "done"
        ]
        print(f'Available skill set: {avail_skill_set}')
        
        # initial task trajectory logging file
        collect_file_path = os.path.join(
            self.text_trajectory_save_dir, 
            str(env.task_name)
        )
        if ".json" in collect_file_path:
            sp = env.task_name.split("/") 
            subdir = "/".join(sp[:-1])
            os.makedirs(
                os.path.join(
                    self.text_trajectory_save_dir, 
                    subdir
                ),
                exist_ok=True
            )
        collect_file_path = collect_file_path.replace(".json", ".txt")
        collect_file = open(collect_file_path, 'w')
        # initial react observation logging 
        init_obs = env.reset(scene_data, is_reflexion=is_reflexion)

        # Get initial skill set from initial observation information 
        print("get initial skil set from initial obaservation !! ")
        skill_set = self.llm_handler.update_skill_set(init_obs)

        nl_inst_info = {
            'nl_inst': env.instruction, 
            # 'task_type': env.task_type,
            'message': None,
            # 'human_attributes': self.get_human_attributes_prompt(env.target_person.attributes)
        }
        # 1. llm reset --> reset user prompt 
        self.llm_handler.reset(nl_inst_info, init_obs)
        
        # logging 
        init_obs_str = init_obs['text']
        instruction_str = f'Your task is to: {env.instruction}'

        collect_file.write(f'{instruction_str}\n')
        # collect_file.write(f'{human_attributes_str}\n')
        collect_file.write(f'{init_obs_str}\n')
        
        print(f'{instruction_str}')
        # print(f'{human_attributes_str}')
        print(f'{init_obs_str}')

        # interactive trajectory collection start
        while True:
            print()
            print('"r" : thnink "a" : act')
            decision = input('Type decision ("r" or "a"): ')
            if decision == 'r':
                reasoning_text = input('Think: ')
                collect_file.write(f'Think: {reasoning_text}\nOK.\n')
            elif decision == 'a':
                action_text = input('Act: ')
                if action_text in ['done', 'failure']:
                    collect_file.write(f'Act: {action_text}')
                    break
                else: # interact with env 
                    # try:
                    action_ret = env.llm_skill_interact(action_text)
                    print(action_ret)
                    self.llm_handler.update_skill_set(action_ret)  
                    #print('react.py action_ret: ', action_ret)
                    obs_text = action_ret['message']
                    print(obs_text)
                    collect_file.write(f'Act: {action_text}\n{obs_text}\n')     
                    # except Exception as e:
                    #     print("Error. Type again.", e)
                    #     # collector is trajectory saving purpose
                    #     # so the error case is not recorded. 
                    #     # untill the action is successfully executed.
            else:
                print(f'Error: Type again. ("r" or "a") your input {decision}')
                pass
        
        # evaluate results 
        goal_results = env.goal_checker.check_goal(env.controller.last_event)
        print("Task Success : ", goal_results)
