# Maze-RL L2RPN - WCCI 2022 Submission

This repository contains the code for running [enliteAI's](https://www.enlite.ai/) 1st place submission
to the [Learning to Run a Power Network (L2RPN)](https://l2rpn.chalearn.org/) - 
[Energies of the Future and Carbon Neutrality](https://codalab.lisn.upsaclay.fr/competitions/5410)
Reinforcement Learning (RL) competition.

For details about our agent as well as background information about grid topology optimization in general
we invite you to check out our paper

>**[Power Grid Congestion Management via Topology Optimization with AlphaZero](https://arxiv.org/pdf/2211.05612.pdf).**\
Matthias Dorfer∗, Anton R. Fuxjäger∗, Kristian Kozak∗, Patrick M. Blies and Marcel Wasserer.\
*RL4RealLife Workshop in the 36th Conference on Neural Information Processing Systems (NeurIPS 2022).*\
(*equal contribution)

*Note that our submission goes beyond the agent described and evaluated in the paper
with respect to grid specific enhancements such as contingency analysis.*

### Overview
* [Running our Submission](#running-submission)
* [Web Demo - AI Assistant Power Grid Control](#web-demo)
* [Further Resources](#resources)
* [About our RL Framework Maze](#about-maze)

<a name="running-submission"></a>
## Running our Submission

To test our submission just follow the steps listed below.

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


You can also find more detailed results in this directory:

    <absolute local dir to the extracted starting kit>/utils/last_submission_results/

![results-preview](https://github.com/enlite-ai/maze-l2rpn-2022-submission/raw/main/results.png)

<a name="web-demo"></a>
## Web Demo - AI Assistant Power Grid Control

The figure below shows the workflow of our
real-time remedial action recommendation [assistant demo](http://grid-demo.enlite.ai/ )
as a concrete example for human-in-the-loop decision making.

![assistant-overview](https://github.com/enlite-ai/maze-l2rpn-2022-submission/raw/main/assistant_overview.png)

The central design principle of our AI assistant is  to support and enhance human decision making
by recommending viable action scenarios along with augmenting information
explaining how these recommendations will most likely turn out in the productive system.
The final decision is left to the human operator to preserve human control.

- The Grid State Observer continuously monitors the current state of the productive power grid.
- Once a non-safe state is encountered the agent starts a policy network guided tree search to discover a
set of topology change action scenarios that are capable of recovering from this critical situation (e.g.,
relieving the congestion).
- A ranked list of topology change candidates – the top results of the tree search – is presented to the
human operator in a graphical user interface for evaluation (testing the impact of a action candidate in
a load flow simulation) and selection. Along with the potential for relieving the congestion other grid
specific safety considerations are taken into account (e.g., a contingency analysis for n-1 stability of
the respective resulting states).
- The human operator evaluates the provided set of suggested topology change actions.
- Once satisfied, he/she confirms the best action candidate for execution on the productive power grid.
- The selected action candidate is applied to the power grid and the resulting state is again fed into the
Grid State Observer and visualized for human operators in the graphical user interface. This closes
the human in the loop workflow.

<a name="resources"></a>
## Further Resources

- Spotlight Talk and [Poster](https://nips.cc/media/PosterPDFs/NeurIPS%202022/58368.png) at the NeurIPS 2022 RL4RealLife Workshop.
- [enliteAI Energy](https://www.enlite.ai/solutions/energy) web page.
- AAIC Energy [Keynote on Powergrid Congesti Management with RL](https://www.youtube.com/watch?v=0zqSAsj_86I).

<a name="about-maze"></a>
## About the RL Framework Maze

![Banner](https://github.com/enlite-ai/maze/raw/main/docs/source/logos/main_logo.png)

[Maze](https://github.com/enlite-ai/maze) is an application-oriented deep reinforcement learning (RL) framework, addressing real-world decision problems.
Our vision is to cover the complete development life-cycle of RL applications, ranging from simulation engineering to agent development, training and deployment.
  
If you encounter a bug, miss a feature or have a question that the [documentation](https://maze-rl.readthedocs.io/) doesn't answer: We are happy to assist you! Report an [issue](https://github.com/enlite-ai/maze/issues) or start a discussion on [GitHub](https://github.com/enlite-ai/maze/discussions) or [StackOverflow](https://stackoverflow.com/questions/tagged/maze-rl).