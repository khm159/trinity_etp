import os
import copy
import json
from .llm_handler import LLMHandler
from .utils import ungroup_objects, ALFRED_PICK_OBJ, AFLRED_OPEN_OBJ, ALFRED_TOGGLE_OBJ, ALFRED_SLICE_OBJ, AI2THOR_NOT_PICKABLE_OBJ, read_txt_file

class TrinityLLMHandler(LLMHandler):
    """base LLM handler for ReAct Agent"""
    def load_prompt(self, nl_inst_info, init_obs_text):
        self.system_prompt = read_txt_file(self.cfg.prompt.system_prompt_path)
        nl_inst = nl_inst_info['nl_inst']

        if "human_attributes" not in nl_inst_info.keys():
            human_attr = None
        else:
            human_attr = nl_inst_info['human_attributes']

        # use only top-1 example 
        ic_ex_encoding = self.ic_ex_select(nl_inst_info)
        self.ic_ex_prompt = ic_ex_encoding['str']

        if human_attr is None:
            # AEDataset 
            if nl_inst_info['message'] == None or nl_inst_info['message'] == '':
                user_prompt = f'### Source Domain:\n{self.ic_ex_prompt}\n### Target Domain:\n{nl_inst}\n{init_obs_text}\n'
        else:
            # HAEDataset 
            if nl_inst_info['message'] == None or nl_inst_info['message'] == '':
                user_prompt = f'### Source Domain:\n{self.ic_ex_prompt}\n### Target Domain:\n{nl_inst}\n{human_attr}\n{init_obs_text}\n'

        return user_prompt
    
    def load_prompt_reflection(self, nl_inst_info, init_obs_text, reflexion_traj):
        self.system_prompt = read_txt_file(self.cfg.prompt.system_prompt_path)
        nl_inst = nl_inst_info['nl_inst']

        if "human_attributes" not in nl_inst_info.keys():
            human_attr = None
        else:
            human_attr = nl_inst_info['human_attributes']

        # use only top-1 example 
        ic_ex_encoding = self.ic_ex_select(nl_inst_info)
        self.ic_ex_prompt = ic_ex_encoding['str']

        if human_attr is None:
            # AEDataset 
            if nl_inst_info['message'] == None or nl_inst_info['message'] == '':
                user_prompt = f'### Source Domain:\n{self.ic_ex_prompt}\n### Target Domain:{reflexion_traj}\nYour task is to: {nl_inst}\n{init_obs_text}\n'
        else:
            # HAEDataset 
            if nl_inst_info['message'] == None or nl_inst_info['message'] == '':
                user_prompt = f'### Source Domain:\n{self.ic_ex_prompt}\n### The information of the person who gave this instruction is as follows:{human_attr}\n### Target Domain:{reflexion_traj}\nYour task is to: {nl_inst}\n{init_obs_text}\n'

        return user_prompt
    
    def get_ic_ex_samples(self):
        return self.system_prompt, self.ic_ex_prompt
    
    def load_predefined_prompt(self, file_path):
        with open(file_path) as file:
            prompt = file.read()
        return prompt
    
    @staticmethod
    def load_ic_ex_str(path):
        file = open(path, "r")
        lines = file.readlines()
        return "".join(lines)

    def ic_ex_select(self, nl_inst_info):
        if self.cfg.agent.ic_ex_select_type == 'rag':
            # 1. extract instruction embedding 
            nl_inst = nl_inst_info['nl_inst']
            nl_inst_embedding = self.sbert.encode(
                nl_inst,
                show_progress_bar = False
            )
            if len(nl_inst_embedding.shape) >1:
                nl_inst_embedding = nl_inst_embedding[0]

            # 2. extract example embeddings
            ic_ex_encode_dir = os.path.join(self.cfg.prompt.ic_ex_dir)
            ic_ex_encode_list = []
            for ic_ex_encode_name in os.listdir(ic_ex_encode_dir):
                ic_ex_encoding=dict()
                ic_ex_encode_path = os.path.join(
                    ic_ex_encode_dir, 
                    ic_ex_encode_name
                )
                ic_ex_str = self.load_ic_ex_str(ic_ex_encode_path)
                ic_ex_feat = self.sbert.encode(
                    ic_ex_str,
                    show_progress_bar = False
                )
                ic_ex_encoding['encoding']=ic_ex_feat
                ic_ex_encoding['path']=ic_ex_encode_path
                similarity = self.cosine_similarity(
                    ic_ex_feat, 
                    nl_inst_embedding
                )
                ic_ex_encoding['similarity'] = similarity
                ic_ex_encoding['str']=ic_ex_str
                ic_ex_encode_list.append(ic_ex_encoding)
            
            sorted_ic_ex_encode_list = sorted(
                ic_ex_encode_list, 
                key=lambda x: x['similarity'], 
                reverse=True
            )
        
        return sorted_ic_ex_encode_list[0]
        
    def update_skill_set(self, obs, is_holodeck=False):
        self.is_init = False # only for updating 
        nl_obs_partial_objs_info = obs['nl_obs_partial_objs_info']
        nl_room_info = obs['nl_room_info']
        room_static_recps = obs['room_static_receps']

        # [1] add initial skills 
        #     done, failure 
        if obs['init_obs']:
            self.is_init = True
        else:
            self.is_init = False
        skill_set = ['done', 'failure']
        
        # [2] Add go to room skills 
        if nl_room_info is None:
            pass
        else:
            for room_name in nl_room_info:
                skill_set.append(f'go to {room_name}')

        # [3] Add static recpes 
        for static_recpe in room_static_recps:
            skill_set.append(f'go to {static_recpe}')

        # [3] Add partial observable skills 
        if nl_obs_partial_objs_info is None:
            pass
        else:
            # print("add partial observable skills")
            for partial_obj_info in nl_obs_partial_objs_info:
                # e.g. ['FloorLamp (1)']
                partial_obj_name = partial_obj_info.split(' ')[0]
                partial_obj = ungroup_objects(partial_obj_info)

                for i in range(len(partial_obj)):
                    skill_set.append(f'go to {partial_obj[i]}')
                    if not is_holodeck:
                        if partial_obj_name in ALFRED_PICK_OBJ:
                            skill_set.append(f'pick up {partial_obj[i]}')
                            skill_set.append(f'put down {partial_obj[i]}')
                    else:
                        if partial_obj_name not in AI2THOR_NOT_PICKABLE_OBJ:
                            skill_set.append(f'pick up {partial_obj[i]}')
                            skill_set.append(f'put down {partial_obj[i]}')

                    if partial_obj_name in AFLRED_OPEN_OBJ:
                        skill_set.append(f'open {partial_obj[i]}')
                        skill_set.append(f'close {partial_obj[i]}')
                        
                    if partial_obj_name in ALFRED_TOGGLE_OBJ:
                        skill_set.append(f'turn on {partial_obj[i]}')
                        skill_set.append(f'turn off {partial_obj[i]}')
                        
                    if partial_obj_name in ALFRED_SLICE_OBJ:
                        skill_set.append(f'slice {partial_obj[i]}')
                        
            if self.is_init:
                self.init_skill_set = list(set(skill_set))

            self.skill_set = skill_set
            self.skill_set = list(set(skill_set))
            self.skill_set.sort()
        
            return self.skill_set

class TrinityPreActLLMHandler(TrinityLLMHandler):
    """LLM handler for PreAct Agent"""
    
    def parse_next_step(self, gen_lines):
        if isinstance(gen_lines, str):
            gen_lines = gen_lines.split("\n")
        elif isinstance(gen_lines, dict):
            content = gen_lines['content']
            gen_lines = content.split("\n")
        step_class = None
        think_start_ind = None
        thought_lines = []
        prediction = ""
        
        has_prediction = False
        for line in gen_lines:
            if "Predict" in line:
                has_prediction = True
                break
            elif "predict" in line:
                has_prediction = True
                break

        if has_prediction:
            is_start = False
            for line in gen_lines:

                if "Predict" in line:
                    prediction += line
                    is_start = True
                    
                elif "predict" in line:
                    prediction += line
                    is_start = True
                    
                if "Think:" in line:
                    is_start = False
                    break
                elif "Think" in line:
                    is_start = False
                    break
                
                if "Act:" in line:
                    continue
                
                if "Act" in line:
                    continue
                
        # [1] extract "think and act"
        is_think=False
        for line_ind, line in enumerate(gen_lines):
            if "OK." in line:
                if not think_start_ind is None:
                    thought_end_ind = line_ind -1
                    thought = "\n".join(thought_lines)
                    step_class = "Think"
                    return step_class, thought, prediction
                else:
                    continue
            if "Think:" in line:
                step_class = "Think"
                line = line.replace("Think:", "").strip()
                think_start_ind = line_ind
                thought_lines.append(line)
                is_think=True
                continue
            elif "Think" in line:
                step_class = "Think"
                line = line.replace("Think", "").strip()
                think_start_ind = line_ind
                thought_lines.append(line)
                is_think=True
                continue
            elif "Act:" in line:
                if not is_think:
                    step_class = "Act"
                    line = line.replace("Act:", "").strip()
                    return step_class, line, prediction
                else:
                    break
            elif "Act" in line:
                if not is_think:
                    step_class = "Act"
                    line = line.replace("Act", "").strip()
                    return step_class, line, prediction
                else:
                    break
            else:
                if not is_think:
                    thought_lines.append(line)
                continue
        
        if not len(thought_lines) == 0:
            thought = "\n".join(thought_lines)
        else:
            thought = ""

        return step_class, thought, prediction
    
    def preact_plan_next_step_think_every_step(self, skill_set, steps):
        print("PreAct Plan nextstep - think every step - ")
        thought = ''
        prediction = ''
        step_class=None
        action_step=None
        is_json_dump_success=False
        has_think_key=False
        has_act_key=False
        has_prediction_key=False
        if 'gpt' in self.model_name:
            ### openAI LLM
            max_try = 5
            for try_ind in range(max_try):
                is_json_dump_success=False
                has_think_key=False
                has_act_key=False
                has_prediction_key=False
                # forcely generate the output to JSON format 

                # generate JSON output 
                gen_result = self.inference_gpt_func_call(self.prompt)
                _gen_result = copy.deepcopy(gen_result)
                try:
                    gen_dict = json.loads(gen_result)
                except:
                    print("Error: JSON format error. Trying again...")
                    continue
                is_json_dump_success=True

                # parse the JSON keys 
                if "think" not in gen_dict.keys():
                    if "Think" not in gen_dict.keys():
                        print("Error: Thought not in the generated result. Trying again...")
                        continue
                    else:
                        thought = gen_dict['Think']
                else:
                    thought = gen_dict['think']
                has_think_key=True
                if "act" not in gen_dict.keys():
                    if "Act" not in gen_dict.keys():
                        print("Error: Act not in the generated result. Trying again...")
                        continue
                    else:
                        act_content = gen_dict['Act']
                else:
                    act_content = gen_dict['act']

                has_act_key=True
                if "predict" not in gen_dict.keys():
                    if "Predict" not in gen_dict.keys():
                        print("Error: Prediction not in the generated result. Trying again...")
                        continue
                    else:
                        prediction = gen_dict['Predict']
                else:
                    prediction = gen_dict['predict']

                has_prediction_key=True
                step_class = "Act"

                if step_class is not None:
                    break
        else:
            NotImplementedError("Not implemented yet")
        
        ### parse parital information if available from the JSON 
        if step_class is None:
            # just return the last result
            if is_json_dump_success:
                if not has_act_key and not has_think_key and not has_prediction_key:
                    # if the LLM is not generate the proper json keys and foramt
                    step_class = "Think"
                    thought = _gen_result.split("\n")[0]
                    if self.use_local_model:
                        self.partial_fail_log = open(self.partial_fail_log_name,"a")
                        self.partial_fail_log.write("================[JSON decode success but no think, act or predict key]===================\n")
                        self.partial_fail_log.write("try index : {}\n".format(try_ind+1))
                        self.partial_fail_log.write(_gen_result+"\n")
                        self.partial_fail_log.close()

                if has_act_key:
                    if "act" in gen_dict.keys():
                        act_content = gen_dict['act']
                        step_class = "Act"
                    elif "Act" in gen_dict.keys():
                        act_content = gen_dict['Act']
                        step_class = "Act"
                    else: # fail to parse the action content
                        act_content = None
                        step_class = "Think"
                        thought = _gen_result.split("\n")[0]
                        if self.use_local_model:
                            self.partial_fail_log = open(self.partial_fail_log_name,"a")
                            self.partial_fail_log.write("================[JSON decode success but no think or act key]===================\n")
                            self.partial_fail_log.write("try index : {}\n".format(try_ind+1))
                            self.partial_fail_log.write(_gen_result+"\n")
                            self.partial_fail_log.close()
                
                if has_think_key:
                    if "think" in gen_dict.keys():
                        thought = gen_dict['think']
                    elif "Think" in gen_dict.keys():
                        thought = gen_dict['Think']
                    else: # fail to parese the thought content 
                        thought = gen_result.split("\n")[0]

                        if self.use_local_model:
                            self.partial_fail_log = open(self.partial_fail_log_name,"a")
                            self.partial_fail_log.write("================[JSON decode success but no think key]===================\n")
                            self.partial_fail_log.write("try index : {}\n".format(try_ind+1))
                            self.partial_fail_log.write(_gen_result+"\n")
                            self.partial_fail_log.close()

                    step_class = "Think"

                if has_prediction_key:
                    if "predict" in gen_dict.keys():
                        prediction = gen_dict['predict']
                        
                    elif "Predict" in gen_dict.keys():
                        prediction = gen_dict['Predict']
                    else:
                        prediction = ""
                        # TODO: Parsing codes for open source models
                        if self.use_local_model:
                            self.partial_fail_log = open(self.partial_fail_log_name,"a")
                            self.partial_fail_log.write("================[JSON decode success but no predict key]===================\n")
                            self.partial_fail_log.write("try index : {}\n".format(try_ind+1))
                            self.partial_fail_log.write(_gen_result+"\n")
                            self.partial_fail_log.close()
            else:
                # just return the last result
                step_class = "Think"
                gen_result = _gen_result.split("\n")[0]

        ## Execution of the plan
        if step_class =="Act":
            print("     >>> Generated Output Action ", act_content)
            # skill furzzy-matching 
            generated_action_embedding = self.sbert.encode(
                act_content,
                show_progress_bar = False
                )
            skill_set_encodings = []
            for skill in skill_set:
                skill_encode_dict = dict()
                skill_embedding = self.sbert.encode(
                    skill,
                    show_progress_bar = False
                )
                similarity = self.cosine_similarity(
                    skill_embedding, 
                    generated_action_embedding
                )
                skill_encode_dict['feat']=skill_embedding
                skill_encode_dict['similarity'] = similarity
                skill_encode_dict['str'] = skill
                skill_set_encodings.append(skill_encode_dict)    
            
            sorted_ic_ex_encode_list = sorted(
                skill_set_encodings, 
                key=lambda x: x['similarity'], 
                reverse=True
            )
            action_step = sorted_ic_ex_encode_list[0]['str']
            print("    >>> Matched Action Step ", action_step)
            
        next_step_info = {
            'next_step_class': step_class, 
            'next_step': action_step,
            'thought': thought,
            'prediction': prediction
        }

        if step_class == "Think":
            # error case, when the LLM is not generate the proper json keys and foramt
            self.prompt += f"Think: {thought}\n"
            if not prediction == "":
                self.prompt += f"Predict: {prediction}\n"
        elif step_class == "Act":
            self.prompt += f"Think: {thought}\n"
            self.prompt += f"Act: {action_step}\n"
            if not prediction == "":
                self.prompt += f"Predict: {prediction}\n"
        else:
            self.prompt += gen_result.strip()+"\n"
            if not prediction == "":
                self.prompt += f"Predict: {prediction}\n"
        return next_step_info

    def preact_plan_next_step(self, skill_set, steps):
        thought = ''
        prediction = ''
        step_class=None

        if 'gpt' in self.model_name:
            ### openAI LLM
            max_try = 5
            for try_ind in range(max_try):
                gen_result = self.inference_gpt(self.prompt)
                _gen_result = copy.deepcopy(gen_result)
                step_class, gen_result, prediction = self.parse_next_step(gen_result)
                if step_class is not None:
                    break
        else:
            NotImplementedError("Not implemented yet")

        if step_class is None:
            # just return the last result
            step_class = "Think"
            gen_result = _gen_result.split("\n")[0]

        ## Execution of the plan
        if step_class =="Act":
            # skill furzzy-matching 
            gen_res_embedding = self.sbert.encode(
                gen_result,
                show_progress_bar = False
                )
            skill_set_encodings = []
            for skill in skill_set:
                skill_encode_dict = dict()
                skill_embedding = self.sbert.encode(
                    skill,
                    show_progress_bar = False
                )
                similarity = self.cosine_similarity(
                    skill_embedding, 
                    gen_res_embedding
                )
                skill_encode_dict['feat']=skill_embedding
                skill_encode_dict['similarity'] = similarity
                skill_encode_dict['str'] = skill
                skill_set_encodings.append(skill_encode_dict)    
            
            sorted_ic_ex_encode_list = sorted(
                skill_set_encodings, 
                key=lambda x: x['similarity'], 
                reverse=True
            )
            gen_result = sorted_ic_ex_encode_list[0]['str']
    
        next_step_info = {
            'next_step_class': step_class, 
            'next_step': gen_result,
            'thought': thought,
            'prediction': prediction
        }
        if step_class == "Think":
            self.prompt += f"Think: {gen_result}\n"
        elif step_class == "Act":
            self.prompt += f"Act: {gen_result}\n"
            if not prediction == "":
                self.prompt += f"{prediction}\n"
        else:
            self.prompt += gen_result.strip()+"\n"
        return next_step_info

class TrinityStateActLLMHandler(TrinityLLMHandler):
    """LLM handler for PreAct Agent"""
    
    def parse_next_step(self, gen_lines):
        if isinstance(gen_lines, str):
            # for Qwen and others 
            gen_lines = gen_lines.split("\n")
        elif isinstance(gen_lines, dict):
            # for llama or pipeline 
            content = gen_lines['content']
            gen_lines = content.split("\n")
        step_class = None
        think_start_ind = None
        thought_lines = []
        prediction = ""
        
        has_prediction = False
        for line in gen_lines:
            if "Predict" in line:
                has_prediction = True
                break
            elif "predict" in line:
                has_prediction = True
                break

        if has_prediction:
            is_start = False
            for line in gen_lines:

                if "Predict" in line:
                    is_start = True
                    
                elif "predict" in line:
                    is_start = True
                    
                if "Think:" in line:
                    is_start = False
                    break
                elif "Think" in line:
                    is_start = False
                    break
                
                if "Act:" in line:
                    continue
                
                if "Act" in line:
                    continue
                
                if is_start:
                    prediction += line
                
        # [1] extract "think and act"
        is_think=False
        for line_ind, line in enumerate(gen_lines):
            if "OK." in line:
                if not think_start_ind is None:
                    thought_end_ind = line_ind -1
                    thought = "\n".join(thought_lines)
                    step_class = "Think"
                    return step_class, thought, prediction
                else:
                    continue
            if "Think:" in line:
                step_class = "Think"
                line = line.replace("Think:", "").strip()
                think_start_ind = line_ind
                thought_lines.append(line)
                is_think=True
                continue
            elif "Think" in line:
                step_class = "Think"
                line = line.replace("Think", "").strip()
                think_start_ind = line_ind
                thought_lines.append(line)
                is_think=True
                continue
            elif "Act:" in line:
                if not is_think:
                    step_class = "Act"
                    line = line.replace("Act:", "").strip()
                    return step_class, line, prediction
                else:
                    break
            elif "Act" in line:
                if not is_think:
                    step_class = "Act"
                    line = line.replace("Act", "").strip()
                    return step_class, line, prediction
                else:
                    break
            else:
                if not is_think:
                    thought_lines.append(line)
                continue
        
        if not len(thought_lines) == 0:
            thought = "\n".join(thought_lines)
        else:
            thought = ""

        return step_class, thought, prediction
    
    def stateact_plan_next_step_think_every_step(self, skill_set, goal_inst):
        thought = ''
        location = ''
        inventory = ''
        step_class=None
        action_step=None

        is_json_dump_success=False
        has_think_key=False
        has_act_key=False
        has_location_key=False
        has_inventory_key=False

        # Add "Goal:"~"
        self.prompt += goal_inst.replace("Your task is to: ","").strip()+"\n"

        if 'gpt' in self.model_name:
            ### openAI LLM
            max_try = 5
            for try_ind in range(max_try):
                is_json_dump_success=False
                has_think_key=False
                has_act_key=False

                # generate JSON output 
                gen_result = self.inference_gpt_func_call(self.prompt)
                _gen_result = copy.deepcopy(gen_result)
                try:
                    gen_dict = json.loads(gen_result)
                except:
                    print("Error: JSON format error. Trying again...")
                    continue
                is_json_dump_success=True

                # parse the JSON keys 
                if "think" not in gen_dict.keys():
                    if "Think" not in gen_dict.keys():
                        print("Error: Thought not in the generated result. Trying again...")
                        continue
                    else:
                        thought = gen_dict['Think']
                else:
                    thought = gen_dict['think']
                has_think_key=True


                if "act" not in gen_dict.keys():
                    if "Act" not in gen_dict.keys():
                        print("Error: Act not in the generated result. Trying again...")
                        continue
                    else:
                        act_content = gen_dict['Act']
                else:
                    act_content = gen_dict['act']
                has_act_key=True

                if "current_location" not in gen_dict.keys():
                    if "current location" not in gen_dict.keys():
                        if "Current_location" not in gen_dict.keys():
                            if "Current_Location" not in gen_dict.keys():
                                print("Error: Prediction not in the generated result. Trying again...")
                                continue
                            else:
                                location = gen_dict["Current_Location"]
                        else:
                            location = gen_dict["Current_location"]
                    else:
                        location = gen_dict['current location']
                else:
                    location = gen_dict['current_location']
                has_location_key=True
                
                
                if "current_inventory" not in gen_dict.keys():
                    if "current inventory" not in gen_dict.keys():
                        if "Current_inventory" not in gen_dict.keys():
                            if "Current_Inventory" not in gen_dict.keys():
                                print("Error: Prediction not in the generated result. Trying again...")
                                continue
                            else:
                                inventory = gen_dict["Current_Inventory"]
                        else:
                            inventory = gen_dict["Current_inventory"]
                    else:
                        inventory = gen_dict['current inventory']
                else:
                    inventory = gen_dict['current_inventory']
                has_inventory_key=True

                step_class = "Act"

                if step_class is not None:
                    break
        else:
            NotImplementedError("Not implemented yet")

        ### parse parital information if available from the JSON 
        if step_class is None:
            # just return the last result
            if is_json_dump_success:
                if not has_act_key and not has_think_key and not has_inventory_key and not has_location_key:
                    # if the LLM is not generate the proper json keys and foramt
                    step_class = "Think"
                    thought = _gen_result.split("\n")[0]

                if has_act_key:
                    if "act" in gen_dict.keys():
                        act_content = gen_dict['act']
                        step_class = "Act"
                    elif "Act" in gen_dict.keys():
                        act_content = gen_dict['Act']
                        step_class = "Act"
                    else: # fail to parse the action content
                        act_content = None
                        step_class = "Think"
                        thought = _gen_result.split("\n")[0]
                
                if has_think_key:
                    if "think" in gen_dict.keys():
                        thought = gen_dict['think']
                    elif "Think" in gen_dict.keys():
                        thought = gen_dict['Think']
                    else: # fail to parese the thought content 
                        thought = gen_result.split("\n")[0]
                    step_class = "Think"

                if has_location_key:
                    if "current_location" in gen_dict.keys():
                        location = gen_dict['current_location']
                    elif "Current_Location" in gen_dict.keys():
                        location = gen_dict['Current_Location']
                    else:
                        location = ""
                
                if has_inventory_key:
                    if "current_inventory" in gen_dict.keys():
                        inventory = gen_dict['current_inventory']
                    elif "Current_Inventory" in gen_dict.keys():
                        inventory = gen_dict['Current_Inventory']
                    else:
                        inventory = ""
            else:
                # just return the last result
                step_class = "Think"
                gen_result = _gen_result.split("\n")[0]

        ## Execution of the plan
        if step_class =="Act":
            # skill furzzy-matching 
            generated_action_embedding = self.sbert.encode(
                act_content,
                show_progress_bar = False
                )
            skill_set_encodings = []
            for skill in skill_set:
                skill_encode_dict = dict()
                skill_embedding = self.sbert.encode(
                    skill,
                    show_progress_bar = False
                )
                similarity = self.cosine_similarity(
                    skill_embedding, 
                    generated_action_embedding
                )
                skill_encode_dict['feat']=skill_embedding
                skill_encode_dict['similarity'] = similarity
                skill_encode_dict['str'] = skill
                skill_set_encodings.append(skill_encode_dict)    
            
            sorted_ic_ex_encode_list = sorted(
                skill_set_encodings, 
                key=lambda x: x['similarity'], 
                reverse=True
            )
            action_step = sorted_ic_ex_encode_list[0]['str']
            
        next_step_info = {
            'next_step_class': step_class, 
            'next_step': action_step,
            'thought': thought,
            'current_location': location,
            'current_inventory': inventory,
        }

        if step_class == "Think":
            # error case, when the LLM is not generate the proper json keys and foramt
            if not location == "":
                self.prompt += f"Current Location: {location}\n"
            if not inventory == "":
                self.prompt += f"Current Inventory: {inventory}\n"
            self.prompt += f"Think: {thought}\n"

        elif step_class == "Act":
            if not location == "":
                self.prompt += f"Current Location: {location}\n"
            if not inventory == "":
                self.prompt += f"Current Inventory: {inventory}\n"
            self.prompt += f"Think: {thought}\n"
            # current inventory 
            self.prompt += f"Act: {action_step}\n"
        else:
            self.prompt += gen_result.strip()+"\n"
            
        return next_step_info

    def preact_plan_next_step(self, skill_set, steps):
        thought = ''
        prediction = ''
        step_class=None

        if 'gpt' in self.model_name:
            ### openAI LLM
            max_try = 5
            for try_ind in range(max_try):
                gen_result = self.inference_gpt(self.prompt)
                _gen_result = copy.deepcopy(gen_result)
                step_class, gen_result, prediction = self.parse_next_step(gen_result)
                if step_class is not None:
                    break
        else:
            NotImplementedError("Not implemented yet")

        if step_class is None:
            # just return the last result
            step_class = "Think"
            gen_result = _gen_result.split("\n")[0]

        ## Execution of the plan
        if step_class =="Act":
            # skill furzzy-matching 
            gen_res_embedding = self.sbert.encode(
                gen_result,
                show_progress_bar = False
                )
            skill_set_encodings = []
            for skill in skill_set:
                skill_encode_dict = dict()
                skill_embedding = self.sbert.encode(
                    skill,
                    show_progress_bar = False
                )
                similarity = self.cosine_similarity(
                    skill_embedding, 
                    gen_res_embedding
                )
                skill_encode_dict['feat']=skill_embedding
                skill_encode_dict['similarity'] = similarity
                skill_encode_dict['str'] = skill
                skill_set_encodings.append(skill_encode_dict)    
            
            sorted_ic_ex_encode_list = sorted(
                skill_set_encodings, 
                key=lambda x: x['similarity'], 
                reverse=True
            )
            gen_result = sorted_ic_ex_encode_list[0]['str']
    
        next_step_info = {
            'next_step_class': step_class, 
            'next_step': gen_result,
            'thought': thought,
            'prediction': prediction
        }
        if step_class == "Think":
            self.prompt += f"Think: {gen_result}\n"
        elif step_class == "Act":
            self.prompt += f"Act: {gen_result}\n"
            if not prediction == "":
                self.prompt += f"{prediction}\n"
        else:
            self.prompt += gen_result.strip()+"\n"
        return next_step_info
  
class TrinityActOnlyLLMHandler(TrinityLLMHandler):     
    """LLM handler for Act Agent"""       

    def act_plan_next_step(self, skill_set, steps):  
        thought = ''
        if 'gpt' in self.model_name:
            ### openAI LLM
            max_try = 5
            for try_ind in range(max_try):
                gen_result = self.inference_gpt(self.prompt)
                step_class, gen_result = self.parse_next_step(gen_result)
                if step_class is not None:
                    break
        else:
            NotImplementedError("Not implemented yet")

        ## Execution of the plan
        # skill furzzy-matching 
        gen_res_embedding = self.sbert.encode(
            gen_result,
            show_progress_bar = False
            )
        skill_set_encodings = []
        for skill in skill_set:
            skill_encode_dict = dict()
            skill_embedding = self.sbert.encode(
                skill,
                show_progress_bar = False
            )
            similarity = self.cosine_similarity(
                skill_embedding, 
                gen_res_embedding
            )
            skill_encode_dict['feat']=skill_embedding
            skill_encode_dict['similarity'] = similarity
            skill_encode_dict['str'] = skill
            skill_set_encodings.append(skill_encode_dict)    
        
        sorted_ic_ex_encode_list = sorted(
            skill_set_encodings, 
            key=lambda x: x['similarity'], 
            reverse=True
        )
        gen_result = sorted_ic_ex_encode_list[0]['str']
    
        next_step_info = {
            'next_step_class': step_class, 
            'next_step': gen_result,
            'thought': thought
        }
        if step_class == "Act":
            self.prompt += f"Act: {gen_result}\n"
        else:
            self.prompt += gen_result.strip()+"\n"
        return next_step_info        

    def parse_next_step(self, gen_lines):
        if isinstance(gen_lines, str):
            # for Qwen and others 
            gen_lines = gen_lines.split("\n")
        elif isinstance(gen_lines, dict):
            # for llama or pipeline 
            content = gen_lines['content']
            gen_lines = content.split("\n")
        step_class = None
        for line in gen_lines:
            if "OK." in line:
                continue
            elif "Act:" in line:
                step_class = "Act"
                line = line.replace("Act:", "").strip()
                return step_class, line
            elif "Act" in line:
                step_class = "Act"
                line = line.replace("Act", "").strip()
                return step_class, line
            elif "act:" in line:
                step_class = "Act"
                line = line.replace("act:", "").strip()
                return step_class, line
            elif "act" in line:
                step_class = "Act"
                line = line.replace("act", "").strip()
                return step_class, line
            else:
                continue
        return step_class, None
