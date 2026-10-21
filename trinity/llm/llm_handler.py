# import guidance
import torch
import copy
import json
import re
from sentence_transformers import SentenceTransformer

from trinity.llm.utils import read_txt_file

GPT_MODELS = [
    'gpt-3.5-turbo-instruct', 
    'gpt-3.5-turbo-0125', 
    'gpt-4-turbo-2024-04-09', 
    'gpt-4o-mini-2024-07-18', 
    'gpt-4o-2024-05-13'
]
import pdb
import os
import pickle
import numpy as np
from numpy.linalg import norm
from openai import OpenAI

class LLMHandler():
    """
    LLM handler using huggingface transformers or OpenAI API.
    """
    def __init__(self, cfg):
        """
        loading the LLM model based on the configuration.
        """
        global GPT_MODELS
        self.cfg = cfg
        self.model_name = cfg.llm.model_name
        self.agent_type = cfg.agent.agent_type
        self.sbert = SentenceTransformer(cfg.llm.sbert_model_name)
        self.system_prompt = read_txt_file(
            self.cfg.prompt.system_prompt_path
        )
        self.use_local_model = False
        self.serve_url =cfg.llm.serve_url

        if 'OpenAI' in self.model_name:
            self.model_name = self.model_name.split('/')[-1]
        else:
            self.model_name = self.model_name

        if self.model_name in GPT_MODELS:
            self.llm = OpenAI()
        else:
            NotImplementedError("LLM model not supported")

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        """Calculate cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

    def reset(self, nl_inst_info, init_obs):
        """reset the LLM model """
        global GPT_MODELS
        self.prompt = self.load_prompt(nl_inst_info, init_obs['text'])

    def reset_reflexion(self, nl_inst_info, init_obs, reflexion_traj):
        """reset the LLM model """
        global GPT_MODELS
        self.prompt = self.load_prompt_reflection(nl_inst_info, init_obs['text'], reflexion_traj)

    def add_obs(self, obs_text):
        self.prompt += f'{obs_text}\n'
    
    def load_prompt_reflection(self, nl_inst, init_obs, reflexion_traj):
        raise NotImplementedError()

    def load_prompt(self, nl_inst, init_obs):
        raise NotImplementedError()
    
    def plan_next_step(self, skill_set, steps=[], goal_inst=None):
        if self.cfg.agent.agent_type == 'react':
            if not self.cfg.agent.think_every_step:
                next_step_info = self.react_plan_next_step(skill_set, steps)
            else:
                next_step_info = self.react_plan_next_step_think_every_step(skill_set, steps)

        elif 'preact' in self.cfg.agent.agent_type : # preact_i, preact_p
            if not self.cfg.agent.think_every_step:
                raise ValueError("Must use think every step for preact agent")
            else:
                next_step_info = self.preact_plan_next_step_think_every_step(skill_set, steps)

        elif self.cfg.agent.agent_type == 'act':
            next_step_info = self.act_plan_next_step(skill_set, steps)

        elif self.cfg.agent.agent_type =='stateact':
            next_step_info = self.stateact_plan_next_step_think_every_step(skill_set, goal_inst=goal_inst)

        elif self.cfg.agent.agent_type =='methodact':
            if not self.cfg.agent.think_every_step:
                next_step_info = self.react_plan_next_step(skill_set, steps)
            else:
                next_step_info = self.react_plan_next_step_think_every_step(skill_set, steps)
        
        else:
            raise NotImplementedError()
        return next_step_info
    
    def llm_inference(self, prompt, system_prompt=None):
        """plane LLM inference"""
        thought = ''
        if 'gpt' in self.model_name:
            ### openAI LLM
            gen_result = self.inference_gpt(self.prompt,system_prompt)
        else:
            NotImplementedError("LLM model not supported")

        return gen_result

    def llm_inference_function_call(self, prompt, system_prompt=None):
        """plane LLM inference"""
        thought = ''
        if 'gpt' in self.model_name:
            ### openAI LLM
            gen_result = self.inference_gpt_func_call(
                self.prompt,
                system_prompt
            )
        else:
            NotImplementedError("LLM model not supported")

        return gen_result  

    def act_plan_next_step(self, skill_set, steps):  
        NotImplementedError()

    def preact_plan_next_step(self, skill_set, steps):
        NotImplementedError()      
    
    def preact_plan_next_step_think_every_step(self, skill_set, steps):
        NotImplementedError()
    
    def stateact_plan_next_step_think_every_step(self, skill_set, goal_inst):
        NotImplementedError()
    
    def extract_and_parse_json_blocks(self, text):
        blocks = re.findall(r'\{.*?\}', text, re.DOTALL)
        parsed_results = []

        for block in blocks:
            json_compatible = block.replace("'", '"')
            parsed_results.append(json_compatible)

        return parsed_results

    def react_plan_next_step_think_every_step(self, skill_set, steps):
        thought = ''
        step_class=None
        action_step=None
        is_json_dump_success=False
        has_think_key=False
        has_act_key=False
        if 'gpt' in self.model_name:
            ### openAI LLM
            # there are no parsing failure when using function calling method of openai
            max_try = 5
            for try_ind in range(max_try):
                is_json_dump_success=False
                has_think_key=False
                has_act_key=False
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
                    thought =  gen_dict['think']
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
                step_class = "Act"
                if step_class is not None:
                    break
        else:
            NotImplementedError("LLM model not supported")

        ### parse parital information if available from the JSON 
        if step_class is None:
            # just return the last result
            if is_json_dump_success:
                if not has_act_key and not has_think_key:
                    # if the LLM is not generate the proper json keys and foramt
                    if self.use_local_model:
                        self.partial_fail_log = open(self.partial_fail_log_name,"a")
                        self.partial_fail_log.write("================[JSON decode success but no think or act key]===================\n")
                        self.partial_fail_log.write("try index : {}\n".format(try_ind+1))
                        self.partial_fail_log.write(_gen_result+"\n")
                        self.partial_fail_log.close()

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
                    if self.use_local_model:
                        self.partial_fail_log = open(self.partial_fail_log_name,"a")
                        self.partial_fail_log.write("================[JSON decode success and has act key but no think key]===================\n")
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
                    step_class = "Think"
                    if self.use_local_model:
                        self.partial_fail_log = open(self.partial_fail_log_name,"a")
                        self.partial_fail_log.write("================[JSON decode success and has think key but no act key]===================\n")
                        self.partial_fail_log.write("try index : {}\n".format(try_ind+1))
                        self.partial_fail_log.write(_gen_result+"\n")
                        self.partial_fail_log.close()
            else:
                # just return the last result
                if self.use_local_model:
                    self.json_dump_fail_log = open(self.json_dump_fail_log_name,"a")
                    self.json_dump_fail_log.write("================[JSON decode failure]===================\n")
                    self.json_dump_fail_log.write("try index : {}\n".format(try_ind+1))
                    self.json_dump_fail_log.write(_gen_result+"\n")
                    self.json_dump_fail_log.close()
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
        }

        if step_class == "Think":
            # error case, when the LLM is not generate the proper json keys and foramt
            self.prompt += f"Think: {thought}\n"
        elif step_class == "Act":
            self.prompt += f"Think: {thought}\n"
            self.prompt += f"Act: {action_step}\n"
        else:
            self.prompt += gen_result.strip()+"\n"

        return next_step_info

    def react_plan_next_step(self, skill_set, steps):
        thought = ''
        if 'gpt' in self.model_name:
            ### openAI LLM
            max_try = 5
            for try_ind in range(max_try):
                gen_result = self.inference_gpt(self.prompt)
                _gen_result = copy.deepcopy(gen_result)
                step_class, gen_result = self.parse_next_step(gen_result)
                print(">>>>>> {}-th try gpt Parsing Result : ".format(try_ind), step_class, gen_result)
                if step_class is not None:
                    break
        else:
            NotImplementedError("LLM model not supported")

        if step_class is None:
            # just return the last result
            self.parsing_fail_log = open(self.parsing_fail_log_name,"a")
            self.partial_fail_log.write("================[free-form gen parsing failure]===================\n")
            self.partial_fail_log.write("try index : {}\n".format(try_ind+1))
            self.partial_fail_log.write("this line treated as think : \n")
            self.partial_fail_log.write(_gen_result+"\n")
            self.partial_fail_log.close()
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
            'thought': thought
        }
        if step_class == "Think":
            self.prompt += f"Think: {gen_result}\n"
        elif step_class == "Act":
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
        think_start_ind = None
        thought_lines = []

        is_think=False
        for line_ind, line in enumerate(gen_lines):
            if "OK." in line:
                if not think_start_ind is None:
                    thought_end_ind = line_ind -1
                    thought = "\n".join(thought_lines)
                    step_class = "Think"
                    return step_class, thought
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
                    return step_class, line
                else:
                    break
            elif "Act" in line:
                if not is_think:
                    step_class = "Act"
                    line = line.replace("Act", "").strip()
                    return step_class, line
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

        return step_class, thought

    def inference_gpt(self, user_prompt, system_prompt=None):
        """Inference for OpenAI GPT models"""
        if system_prompt is None:
            system_prompt = self.system_prompt
        completion = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        return completion.choices[0].message.content

    def inference_gpt_func_call(self, user_prompt, system_prompt=None):
        """Inference for OpenAI GPT models"""
        if system_prompt is None:
            system_prompt = self.system_prompt
        completion = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content