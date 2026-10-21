# Trinity Benchmark Installation Instructions

### Get AI2THOR Binaries
Trinity uses dual AI2THOR binaries for environment simulation.
Make sure to run the following script to set up the necessary binaries:
```bash
bash set_binary.sh
```

### Create and Activate Conda Environment
```bash
conda create -y -n trinity-bench python=3.10
conda activate trinity-bench
```

### Initialize Third-Party Submodules
```bash
git submodule sync
git submodule update --init --recursive
```

### Install Dependencies and Requirements
[Holodeck Installation Reference](https://github.com/allenai/Holodeck?tab=readme-ov-file#installation)
```bash
pip install -e ./third_party/Holodeck/
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+8524eadda94df0ab2dbb2ef5a577e4d37c712897
pip ineall -e .
```

### Setup objathor assets
[Holodeck data download Reference](https://github.com/allenai/Holodeck?tab=readme-ov-file#data)
```bash
# Download data
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
python -m objathor.dataset.download_assets --version 2023_09_23
python -m objathor.dataset.download_annotations --version 2023_09_23
python -m objathor.dataset.download_features --version 2023_09_23
# Create a symbolic link for Holodeck assets in the project directory
ln -s ~/.objathor-assets/2023_09_23/assets .assets
```

### Run Benchmark
```bash
export OPENAI_API_KEY="your api key"
python evaluate.py --config-name config_trinity_act_HAE_L1_gpt4o-mini_DA
```

