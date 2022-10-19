# Maze-RL L2RPN - WCCI 2022 Submission

This repository contains the code for running [enliteAI's](https://www.enlite.ai/) 1st place submission
to the [Learning to Run a Power Network (L2RPN)](https://l2rpn.chalearn.org/) - 
[Energies of the Future and Carbon Neutrality](https://codalab.lisn.upsaclay.fr/competitions/5410)
Reinforcement Learning (RL) competition.

The code in this repository builds on our [RL framework Maze](https://github.com/enlite-ai/maze).

## Running our Submission

To test our submission simply follow the steps listed below.

**1. Download and Extract the Getting Started Kit**

You can download it from [here](https://codalab.lisn.upsaclay.fr/competitions/5410#learn_the_details-get_starting_kit).

**2. Download and Extract the Model Data**

You can download model weights and parameters from [here](https://github.com/enlite-ai/maze-l2rpn-2022-submission/releases/download/WCCI_2022_submission/experiment_data_l2rpn_2022.zip).

Next, copy the content of this archive into the directory `submission/experiment_data`.

    submission
    ├── experiment_data
    │   ├── .hydra
    │   ├── redispatching_CA_KNN 
    │   ├── obs_norm_statistics.pkl
    │   └── ...

**3. Check our Submission**

To check our submission on the 52 chronics of the local validation set run the following docker command
(this should take approximately one hour depending on your machine):

    docker run -it \
    -v <absolute local dir to the extracted starting kit>:/starting_kit \
    -v <absolute local dir to this repo>:/submission \
    -w /starting_kit bdonnot/l2rpn:wcci.2022.1 \
    python check_your_submission.py --model_dir /submission/submission

**4. Inspect Results**

Once complete the agent should achieve the following score:

          0             1
    0     score     70.332381
    1  duration   3009.441725


You can also find more detailed results in this  directory:

    <absolute local dir to the extracted starting kit>/utils/last_submission_results/

![results-preview](https://github.com/enlite-ai/maze-l2rpn-2022-submission/raw/main/results.png)

## About the RL Framework Maze

![Banner](https://github.com/enlite-ai/maze/raw/main/docs/source/logos/main_logo.png)

[Maze](https://github.com/enlite-ai/maze) is an application-oriented deep reinforcement learning (RL) framework, addressing real-world decision problems.
Our vision is to cover the complete development life-cycle of RL applications, ranging from simulation engineering to agent development, training and deployment.
  
If you encounter a bug, miss a feature or have a question that the [documentation](https://maze-rl.readthedocs.io/) doesn't answer: We are happy to assist you! Report an [issue](https://github.com/enlite-ai/maze/issues) or start a discussion on [GitHub](https://github.com/enlite-ai/maze/discussions) or [StackOverflow](https://stackoverflow.com/questions/tagged/maze-rl).