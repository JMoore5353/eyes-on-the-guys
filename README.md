
![73B54EBB-BEBE-4324-AC6A-F803822598A8](https://github.com/user-attachments/assets/9c6d3f6c-014e-4703-84a7-8fd3addcb02a)

https://github.com/user-attachments/assets/9d24bc4f-419d-4aaa-8c04-d8ae6eea8c84


## How To Run 

The only dependency is [Podman](https://podman.io/docs/installation) or [Docker](https://docs.docker.com/engine/install/ubuntu/).
Although Docker is more well known, Podman is more secure as it runs in user-space and not root-space like Docker.
Podman also has a much simpler installation process, so we recommend using Podman.

To run the sim, just run the `build-and-run.sh` bash script.
This will build the included Dockerfile and then run it with the permissions and configuration necessary to display GUIs.
This will take some time, so please be patient (a few minutes to build, about 30 seconds to start flying once the visualizer has loaded).

## Problem Setup
The eyes on the guys (EOTG) problem is a search and rescue problem.
A group of $n$ searchers (guys) are exploring an environment and accumulating information.
A relay agent (fixed-wing UAV) is visiting the searchers to get "eyes on the guys", receive the information gathered by the searchers and relay the information gathered by the other searchers.
The relay agent must decide which searcher to visit and when, which lends itself to a decision process.
Specifically, the agent must weigh the distance to each searcher, the amount of new information they may have, how much information the relay agent has shared so far and the time since visiting that searcher compared to other searchers.
Flying a simple circuit of the searchers could lead to poor performance in some circumstances.
For example, two groups of searchers that are far from one another would cause the relay agent to spend large periods of time traveling between groups.
It could be better to share information within the group for a period of time, accumulating information and then go to the second group.
Online planning and estimating the utility of a given action allows for more dynamic behavior as described above.

<!-- Diagram of searchers and agents -->

We chose to compare the different online planning methods discussed, namely forward search, branch-and-bound and Monte Carlo tree search.
Each method used the same reward function, transition model and scaling parameters.
The comparison helped us to explore the trade-offs of optimality and speed.

### State and Reward Function
The state of the system is the position of the relay agent $\boldsymbol{p}_r$ , the system's shared information matrix $I$, the position of the searchers $\boldsymbol{p}_i | \forall i \in [0,n]$ and time since last visit to the searchers $t_i$.

The reward function is as follows,

$$ R(s') = \lVert{I}\rVert_F - l(s') - \sum_{i=0}^n t_i $$

where $l(s')$ is the length to the proposed searcher from the current searcher.

Upon the relay agent getting to the next searcher, it updates the shared information, distances to adjoining agents and time since visiting each agent.

### Transition Model
The transition model is taken to be deterministic, meaning the probability of going to the intended state is 1.
This makes sense for an agent such as a UAV.
This does ignore the possibility of an opportunistic overflight of another searcher on the way to an intended searcher, but we exclude this case from our simulation.
Furthermore, it is possible the reward function would have the relay agent visit that agent along the course anyway, since the path length penalty would be minimized by doing such.

## Implemented Solutions
We implemented three online planners to compare optimality and runtime.

### Forward Search
Forward search searches over all possible choices to a depth $d$.
At the leaf node, it completes $m$ rollouts and averages the reward to assign the value of the leaf state.
It then recursively updates preceding states with rewards and yields the sequence of searchers to visit that maximizes the utility.
This solution yields the optimal solution for a given depth but can often be intractable depending on the number of searchers and depth.
The number of agents affects the branching factor causing the runtime to explode. 
Depth similarly affects runtime; these effects are analyzed in the analysis function.
This method was used as a ground truth against which the other methods were compared.

<img width="1014" height="597" alt="Forward Search Eyes On The Guys" src="https://github.com/user-attachments/assets/ded11149-7d88-4f79-91e2-64d0ac2ad2d5" />

### Branch-and-Bound
Branch-and-bound attempts to find the optimal solution while avoiding the necessity of searching over the entire state space.
This is done by calculating an upper bound on the total possible future reward for any given state, and pruning any search directions whose maximum bound is lower than either the current best solution or the best reward lower bound on any partial solution.
This method relies on being able to determine a good upper bound that results in substantial pruning, as there is no generalized method to do so and a bound that is too conservative will not prune many branches.

In our implementation rollouts were not used, as very deep searches were found to be adequately fast and instead of implementing a rollout, we simply ran the search to be the same depth as the rollout would have been.

#### Upper Bound
Our upper bound for this problem was chosen to prioritize speed, enabling very deep searches to be done in a few milliseconds but at the cost of optimality, as the bound sometimes will prune the optimal solution.

The bound is calculated with the following algorithm, given the current partially determined path and problem state:
1. Visit states in a greedy manner so as to maximize the total distance of the path. Avoid revisiting states that have already been visited, unless all states have been visited in which case the 'visited' states are reset to unvisited and the algorithm continues until search depth.
2. Calculate the total reward of the path WITHOUT time or distance penalties.
3. Add this reward to the current reward of the partial solution.
4. Return this value as the upper bound.

The motivation behind this bound comes from the structure of our reward function.
In the absence of time or distance penalties, the information gain of the system would give larger rewards for longer paths, as longer paths allow more time for information to be accumulated.
It would still encourage avoiding frequent visits to the same searcher however, as searchers with longer times since their last visit would yield larger rewards.
Hence, a 'longest sequential path' is a reasonable approximation of the non-penalized solution.
We then rely on the fact that the actual reward function is penalized, unlike our bound, to ensure we do not prune too many good solutions.

### Monte Carlo Tree Search

## Expected Behavior and Analysis

### Expected Behavior

<img width="1118" height="1034" alt="live_plots" src="https://github.com/user-attachments/assets/77be526f-3dfd-44ab-a080-b10b7b259909" />

When running this repo you will see a UAV in a simulation environment take off and fly toward one of the searchers. 
A plot will update dynamically comparing the suggested sequence of agents to visit by each solution.
In the simulation environment you will see green lines between each agent as the proposed solution given by MCTS.
There will be a number above each agent which indicates the order to visit the searchers starting with 0 up to $d$.
The relay agent will continue to visit the agents periodically until the program is cancelled.

The relay agent only plans a new sequence for each solution type when it reaches a new agent.
The dynamic plots draw a line from the agent's current position to the next suggested searcher despite travelling to a different agent.

### Analysis
Below we present the results of running the solution methods on the same system.
We compare the runtime versus depth and reward versus depth.

<!-- Plots!!!!!!!! -->
