## Install environment

Create a conda environment

`conda create -n lwm python=3.9`

Install dependencies with pip

`pip install -r requirements.txt`

## Train world models

Change directory

`cd world_model`

Run the training script

`bash scripts/train_wm.py ${MODEL_NAME}`

where `${MODEL_NAME}` is one of `none` (observational, no language), `standardv2` (standard Transformer), `direct` (GPT-hard attention), `emma` (our proposed model), `oracle` (oracle semantic-parsing).

To interact with a trained world model, run:

`bash scripts/play_wm.sh ${MODEL_NAME}`

You can change the `game_id` in `play.py` to visualize a different game.

## Application experiments






