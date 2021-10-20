# The Implementation of a modular Framework for Reinforcement Learning with neuro-evolutional Neural Networks

This repository details the use of a modular framework for Reinforcement Learning, including basic Tabular Q-Learning, DQN and NEAT for Neuro-Evolutional purposes. This Framework was developed under supervision of Prof. Jan Bauer for the use on the Hochshule Karlsruhe P!X3LFLUX Servers to make it simple to incorporate Reinforcement Learning and Neuro-evolution in other Projects and to give a good baseline into Reinforcement Learning.

## Directory

- [Getting Started](#Getting-Started)
    - [List of Dependencies](#list-of-dependencies) 
    - [PIXELFLUX Servers](#PIXELFLUX-Servers)
    - [Installing](#installing)
- [Details about the Framework](#details-about-the-framework)
- [About Reinforcement Learning](#about-reinforcement-learning)
- [About Open AI Gym](#about-open-ai-gym)
    - [Open Ai Gym General Interaction](#Open-Ai-Gym-General-Interaction) 
- [Tabular Q-learning](#tabular-q-learning)
- [Deep Q-learning](#deep-q-learning)
- [NEAT(NeuroEvolution of Augmenting Topologies)](#neat)
    - [Encoding](#encoding)
    - [Mutation](#mutation)
    - [Crossover](#crossover)
    - [Species](#species)
    - [Config File](#config-file)
- [Built With](#built-with)
- [License](#license)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [Future Additions](#future-additions)

## Getting Started

These instructions will get you a copy of the project up and running on the PIXELFLUX Servers.

### List of Dependencies

* `pip install numpy`
* `pip install gym[all]`
* `pip install matplotlib`

For Inline Rendering of the Environment
* `sudo apt-get install xvfb`
* `python -m pip install pyvirtualdisplay`

For Neural Networks and NEAT
* `pip install torchvision`
* `pip install neat-python`

### PIXELFLUX Servers

Start off by logging into the P!X3LFLUX Servers either [redflux](https://eit-redflux.hs-karlsruhe.de) or [greenflux](https://eit-greenflux.hs-karlsruhe.de) to create your own docker container. Once on one of the servers open a terminal window and run `bash ./scripts/startup.sh` to connect to your home drive of the University and to update the required proxy settings.

#### Installing

To get the necessary files onto your docker container go to the preferred directory in a *new terminal*(to update the proxy settings) and execute:
```
git clone https://github.com/PIX3LFLUX/Neuroforcement.git
```
You could also download the repository manually and add the files to your directory.

Once you have cloned the repository you can open the Examples.ipynb and get started with Reinforcement Learning using the Framework. Make sure to set the Kernel to Python 3 (EITB712M) in Jupyter, as this has most of the required packages installed already.

## Details about the Framework

The Project is built as follows: 
* Utilizes Open AI Gym to simulate environments and some additional utility in the **RL_Environments.py**
* Agents are represented by the ways to solve the environments in this case (Tabular Q-Learning, DQN and NEAT) these are run in Juyter Notebook accessing the **Agents.py** for their Models, Parameters and Structures
* Visualizing different information through the **Visualization.py**, including rendering the Environment outputs on the server through the use of a virtual display and plotting information about the reward that the agent receives per episode

These 3 Python files are in the Modules Folder and represent the Framework. The following diagram represents the general interactions of the different modules in the framework. The Examples Jupyter Notebook plays the role of the Main Script in the diagram.

<p align="center">
<img src="https://github.com/PIX3LFLUX/Neuroforcement/blob/main/Media/Framework_Interaction.PNG" width="700">
</p> 

## About Reinforcement Learning
Reinforcement Learning (RL) is a subcategory of machine learning. In its simplest form RL is about teaching an Agent to choose an action in an environment to maximize a scalar reward over time.
The most important concepts and quantities for RL are:

* **Agent**: the part of the program that interacts with the environment and learns
* **Environment**: everything surrounding the agent
* **Reward**: a scalar feedback signal that indicates the quality of the current transition
* **Actions**: the way that agents can interact with the environment
* **State**: every situation encountered in the environment
* **Episode**: a collection of states from initial-state to terminal-state
* **Q-value**: the value of a state-action pair, representing the estimate of the reward the agent thinks it will receive from that transition
* **Policy**: a guide for the agent to choose actions
* **Exploration**: not following the currently best though action, but doing something random in order to gather more information about potentially better rewards

Generally, the Reinforcement Learning Problem can be described as:
An Agent "explores" an environment using a Trial-and-Error approach to find actions that give it the highest reward, which it then tries to maximize. A typical cycle for an Agent would be: first observe the current state, **S0**, then decide to take an action, **A0**, based on past experience or in the interest of exploration. After the action has been performed the agent receives feedback from the environment in the form of a reward, **R1**, and next state, **S1**. This gives the agent one experience set (**S0**, **A0**, **R1**, **S1**). This interaction of gathering one experience set is depicted below. The agent does this until it reaches a terminal state. The collection of states from initial state to terminal state is then referred to as an episode.

<p align="center">
<img src="https://github.com/PIX3LFLUX/Neuroforcement/blob/main/Media/Agent_Environment_Interaction.PNG" width="600">
</p> 

This interaction can be modelled by a Markov Decision Process(MDP). MDP is defined as a tuple of a state-space and a transition function or transition probability. States need to fulfill the Markov Property to be solved, in other words the current state does not depend on previous state, meaning the future is independent of the past given the present

## About Open Ai Gym

* [Open AI Gym](https://gym.openai.com/docs/)

Open Ai Gym provides multiple environments for agents to solve, these include very simple text rendered problems (so called toy text), classic control (CartPole, MountainCar) and up to Atari games and Robotic Control. For this Project the Environments classified under "Toy text", "Classic control" and "Box2D" will be used as a starting point. The advantage of using Open Ai Gym is, that all problems are setup in a similar way as to how the state and reward are extracted and how the agent interaction happens. Meaning that the basic examples help you understand the more complex problems and the dynamics of the problems become familiar.

### Open Ai Gym General Interaction 

Example of one Episode where the Agent can take a maximum of 200 random actions in the CartPole Environment:
```python
env = gym.make('CartPole-v0')

state = env.reset()
for j in range(200):
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    if done:
        break 
env.close()
```
The most important function from Open Ai Gym are displayed here. The `gym.make()` function creates an environment specified by the input, the list of available environments can be found on the Open Ai gym Website. Before any new episode the `env.reset()` function needs to be called to reset the environment back to its initial state, which it also returns so it is assigned to state. The `env.action_space.sample()` function picks a random action from the action space of the environment. So, at this point the action is chosen at random, however this is what our agent needs to determine later on when we introduce it. With the `env.step(action)` function the chosen action is executed in the environment and returns the next state, the reward and a done flag to determine whether the episode is terminal. These functions get executed in a loop until the done flag is set, after which the environment is closed with the `env.close()` function call.

To extend this to training sets where multiple episodes get evaluated a second loop is added which counts the number of episodes.

## Tabular Q-Learning

A method for solving these kinds of problems is called Q-learning. Or more specifically one-step Q-learning. The goal of the agent is to find the optimal action-value function(Q-Function), meaning to find the function that has as input a state and then outputs the action which leads to the most reward. In Q-learning the Q-Function is found by repeatedly interacting with the environment and updating the estimates for the state-action pairs it has seen. This update of the Q-Value is done for each step with this formula Q(S<sub>t</sub>, A<sub>t</sub>) = Q(S<sub>t</sub>, A<sub>t</sub>) +  &alpha;(R<sub>t+1</sub> + &gamma;max<sub>a</sub>Q(S<sub>t+1</sub>, a)-Q(S<sub>t</sub>, A<sub>t</sub>)). Which if done enough leads to the Q-Function. In Tabular Q-Learning the Q-Function is stored as a table which has all available actions as columns and the states as rows. Meaning that given a state and an action a Q-Value can be read from the table. The table entries (or Q-values) are update with the formular as stated above. Given enough transition sets this Q-Table can represent the optimal Q-Function.

<p align="center">
<img src="https://github.com/PIX3LFLUX/Neuroforcement/blob/main/Media/Tab_Q_learning.PNG" width="700">
</p> 

## Deep Q-Learning

When the environment's state space becomes too large to represent the optimal Q-function in the form of a Table a different solution must be considered. In deep Q-learning a neural network is taken to approximate the Q-function. The neural network has as input the state representation and as output the different actions, that the agent can take. The Reward is used in training for back-propagation to update the weights and biases of the neural network to provide a better estimate of the Q-function.

<p align="center">
<img src="https://github.com/PIX3LFLUX/Neuroforcement/blob/main/Media/Deep_Q_learning.PNG" width="700">
</p> 

## NEAT

NEAT (NeuroEvolution of Augmenting Topologies) is a method for modifying neural network weights, topologies, or ensembles in order to learn a specific task. Evolutionary computation is used to search for network parameters that maximize a fitness function that measures performance in a specific task.

<p align="center">
<img src="https://github.com/PIX3LFLUX/Neuroforcement/blob/main/Media/NEAT.PNG" width="700">
</p> 

### Encoding

NEAT uses direct encoding in form of a Genome to represent network structures. A Genome has two sets of genes, one that represents all the different nodes in the network and one that describes the connections of the nodes. The connection genome specifies between which two nodes it occurs, the weight of the connection, whether or not it is enabled, and a so-called innovation number.

<p align="center">
<img src="https://github.com/PIX3LFLUX/Neuroforcement/blob/main/Media/NEAT_Encoding.png" width="600">
</p>

### Mutation

To evolve the neural network structure different mutations can occur. Either existing connections can be changed or new connections or nodes can be added. In the diagram below the cases pf adding a new connection or node is described. When a new node is added it is placed between two nodes that are already connected. This means the previous connection must be disabled; however, it is still represented in the genome.

<p align="center">
<img src="https://github.com/PIX3LFLUX/Neuroforcement/blob/main/Media/NEAT_Mutation.png" width="600">
</p>

### Crossover

The diagram below shows how two different networks are crossed over to form offspring.

<p align="center">
<img src="https://github.com/PIX3LFLUX/Neuroforcement/blob/main/Media/NEAT_Evolution.png" width="600">
</p>

### Species

Speciation splits up the population into several species based on the similarity of topology and connections. The different individuals only need to compete with the part of the population that is classified under its species. This allows for weaker performing structures to be explored more before they are excluded, which serves to preserve innovation.

### Config File

The parameters that describe how the evolution of the algorithm perform are set in a config file. Example parameters that are defined here are: 

* Fitness criterion 
* Fitness threshold
* Population size
* Inputs
* Outputs
* Number of Hidden Layers

## Built With

* [Open AI Gym](https://gym.openai.com/docs/) - Used to Simulate Environments
* [Pytorch](https://pytorch.org/docs/stable/index.html) - Used for the Neural Networks
* [Neat Python](https://neat-python.readthedocs.io/en/latest/neat_overview.html) - Used for the NEAT Implementation
* [Matplotlib](https://matplotlib.org/stable/contents.html) - Used to Plot Data and Results

## Authors

* **Jörn Diemer** - *University of Applied Science Karlsruhe*

## License

GNU General Public License v3.0

## Acknowledgments

* Sutton, R.S. and Barto, A.G.: Reinforcement learning - an introduction. Adaptive computation and machine learning. MIT Press, 1998. ISBN 978-0-262-19398-6. Available at: [Worldcat](http://www.worldcat.org/oclc/37293240)
* [NEAT-Python](https://neat-python.readthedocs.io/en/latest/)
* [Open AI Gym](https://gym.openai.com/)
* Stanley, K.O. and Miikkulainen, R.: Evolving Neural Networks through Augmenting Topologies [NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
* Kansal, S. and Martin, B.: Reinforcement Q-Learning from Scratch in Python with OpenAI Gym [Learndatasci.com](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
* Paszke, A.: REINFORCEMENT LEARNING (DQN) TUTORIAL [Pytorch.org](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* Heidenreich, H.: NEAT: An Awesome Approach to NeuroEvolution [Towardsdatascience.com](https://towardsdatascience.com/neat-an-awesome-approach-to-neuroevolution-3eca5cc7930f)

## Future Additions
* Adapt the Framework to work with more Open Ai Gym environment, especially in the Case of DQN with Pixel Data, this would require writing functions to extract the pixel data
* Do some Parameter Optimization on the Environments that are already working in regards to Neural Network Structure/Optimizers/Replay Memory etc.
* Using vispy to do the rendering in jupyter to utilize the gpu, vispy installation and usage needs to be looked into
* Writing a NEAT Problem that uses Pixel Data to solve problems, which is not in place yet as rendering the environment for each action that NEAT takes would take much too long, especially considering that NEAT is running on the CPU
* Using [Hiddenlayer](https://github.com/waleedka/hiddenlayer) to draw Neural Networks
* Using [Pureples](https://github.com/ukuleleplayer/pureples) and [simondlevy neat-gym](https://github.com/simondlevy/neat-gym) to show NEAT data and results better

