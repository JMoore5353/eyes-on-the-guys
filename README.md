# Eyes On The Guys

<!-- Sizzle Video here? -->

## How To Run 

BRANDON INSTRUCTIONS GO HERE?

## Problem Setup
The eyes on the guys (EOTG) problem is a search and rescue problem.
A group of $n$ searchers (guys) are exploring an environment and accumulating information.
A relay agent (fixed wing UAV) is visiting the agent to get "eyes on the guys", receive the information gathered by the searcher and relay the information gathered by the other searchers.
The relay agent must decide which searcher to visit and when, which lends itself to a decision process.
Specifically, the agent must weigh the distance to each searcher, the amount of new information they may have, how much information the relay agent has shared so far and the time since visiting that searcher compared to other searchers.
Flying a simple circuit of the searchers could lead to poor performance in some circumstances.
For example, two groups of searchers that are far from one another would cause the relay agent to spend large periods of time traveling between groups.
It could be better to share information within the group for a period of time, accumulating information and then go to the second group.
Online planning and estimating the utility of a given action allows for more dynamic behavior like described above.

<!-- Diagram of searchers and agents -->

We chose to compare the different online planning methods discussed, namely forward search, branch-and-bound and Monte Carlo tree search.
Each method used the same reward function, transition model and scaling parameters.
The comparison helped us to explore the trade offs of optimality and speed.

### State and Reward Function
The state of the system is the position of the relay agent $\boldsymbol{p}_r$ , the system's shared information matrix $I$ , the position of the searchers $\boldsymbol{p}_i | \forall i \in [0,n]$ and time since last visit to the searchers $t_i$.

The reward function is as follows,

$$ R(s') = \lVert{I}\rVert_F - l(s') - \sum_{i=0}^n t_i $$

where $l(s')$ is the length to the proposed searcher from the current searcher.

Upon the relay agent getting to the next searcher, it updates the shared information, distances to adjoining agents and time since visiting each agent.

### Transisiton Model
The transition model is taken to be deterministic, meaning the probability of going to the intended state is 1.
This makes sense for an agent such as a UAV.
This does ignore the possibility of an opportunistic overflight of another searcher on the way to an intended searcher, but we exclude this case from our simulation.
Furthermore, it is possible the reward function would have the relay agent visit that agent along the course anyway, since the path length penalty would be minimized by doing such.

## Implemented Solutions
We implemented three online planners to compare optimality and run time.

### Forward Search
Forward search searches over all possible choices to a depth $d$.
At the leaf node, it completes $m$ rollouts and averages the reward to assign the value of the leaf state.
It then recursively updates preceeding states with rewards and yields the sequence of searchers to visit that maximizes the utility.
This solution yields the optimal solution for a given depth but can often be intractable depending on the number of searchers and depth.
The number of agents effects the branching factor causing the run time to explode. 
Depth similarly effects depth, these effects are analyzed in the analysis fuction.
This mehtod was used as a ground truth against which the other methods were compared.


<!-- Diagram forward search -->

### Branch-and-Bound

### Monte Carlo Tree Search

## Expected Behavior and Analysis

### Expected Behavior

<!-- Screenshot of RViz and MPPP -->

When running this repo you will see a UAV in a simulation environment takeoff and fly toward one of the searchers. 
A plot will update dynamically comparing the suggested sequence of agents to visit by each solution.
In the simulation environment you will see green lines between each agent as the proposed solution given by MCTS.
There will be a number above each agent which indicates the order to visit the searchers starting with 0 up to $d$.
The relay agent will continue to visit the agents periodically until the program is canceled.

### Analysis
Below we present the results of running the solution methods on the same system.
We compare the run time versus depth and reward versus depth.

<!-- Plots!!!!!!!! -->
