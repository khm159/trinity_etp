# Trinity Configuration 

```yaml
name: Act_HAE_L1_FA_gpt4o-mini # expriment name (outputs/experinemt name for log)

defaults:
  - hydra: default.yaml

out_dir: ${hydra:run.dir}

# prompt-related configurations
prompt:
  splitter: "" # not used 
  system_prompt_path: 'resource/system_prompt/act_system_with_rules_gpt.txt'
  ic_ex_dir: 'resource/example_dir/Act_HAE_L1_example'
  # system_prompt_path : The system prompt path for each agent.

  # ic_ex_dir : eample directory. 
  # [1] AE setting 
  #    ic_ex_dir should set to AE example of each agent 
  #    (e.g. resource/example_dir/Act_AE_example)
  
  # [2] NA (Naive Adaptation) setting 
  #    ic_ex_dir should set to AE example of each agent 
  #    (e.g. resource/example_dir/Act_AE_example)
  
  # [3] FA (Few-Shot Adaptation) setting
  #    ic_ex_dir should set to HAE example of each agent for each levels.
  #    (e.g. HAE-L3-FA-Act agent : resource/example_dir/Act_HAE_L3_example)

# agent-related configurations
agent: 
  agent_type: 'act'  # agent type 
  ic_ex_select_type: 'rag' 
  max_steps: 60 # maximum action step 
  max_decisions: 99 # maximum think step (not used)
  think_every_step: False # only True for react-im/preact/state

# llm-related configurations 
llm:
  model_name: 'gpt-4o-mini-2024-07-18' # gpt model name
  serve_url: '' # not used
  sbert_model_name: 'all-MiniLM-L6-v2' # sentence embedding model for skill matching
  use_accelerate_device_map: True  
  load_in_8bit: False # not used
  random_seed: 0 # random seed 
  device: 'cuda' # device
  openai_api_key: 'your_key' # openai api key (not used)
  skill_selection_mode: 'fuzzy_matching' # only support fuzzy_matching 

# dataset-related configurations 
trinity:
  dataset_path: 'dataset/HAE_Dataset' 
  train_set: 'train/L1' 
  eval_set: 'val/L1'
  is_hae_dataset: True
  task_goal_template_path: 'dataset/HAE_templates'
  # dataset_path : dataset directory path 
  # train_set : train dataset directory path under dataset_path
  # eval_set  : val dataset directory path under dataset_path 
  # is_hae_dataset : if run HAE dataset, should set to True 
  #                  if not, the assets are not loaded at all.
  # task_goal_template_path : acceptable / prohibited object list 
  #                           template for HAE dataset 

  # If you want to switch the complexity levels 
  # then you need to the following steps 
  # [1] switch trinity.train_set and trinity.eval_set to other levels 
  #   > If you want to run the Level 2 
  #    trinity.train_set : 'train/L1' --> trinity.train_set : 'train/L2' 
  #    trinity.eval_set : 'val/L1' --> trinity.eval_set : 'val/L2' 
  #   > if you run FA setting, then you also should change the example dir
  #    prompt.ic_ex_dir: 'resource/example_dir/Act_HAE_L1_example' to 
  #    prompt.ic_ex_dir: 'resource/example_dir/Act_HAE_L2_example'
   

# ai2thor configurations
ai2thor:
  screen_width: 224  # ai2thor binary resolution
  screen_height: 224 # ai2thor binary resolution
  fov: 90 # fov
  x_display: 0 # x-display (currently not supported)
  is_agent_cam_rgb_frame_save: True
  is_agent_cam_depth_frame_save: False
  is_agent_cam_sementic_mask_frame_save: False
  is_agent_cam_instance_mask_frame_save: False
  is_agent_cam_2d_bbox_frame_save: False
  is_agent_cam_3d_bbox_frame_save: False
  is_topview_cam_rgb_frame_save: True
  is_agent_view_cam_rgb_frame_save: True
  visualize_agent_view_frame: False
  visualize_top_view_frame: True
  cam_frame_save_path: 'trinity_exp_img_dir' # agent cemara image dir 

log:
  save_path: 'work_dir/Act/HAE_L1_FA/gpt4o-mini' # save dir for the trajectories json
```