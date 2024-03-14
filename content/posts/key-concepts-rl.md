---
title: "Key Concepts In (Deep) Reinforcement Learning"
date: 2024-03-12T12:55:08+08:00
draft: false
---

Reinforcement Learning (RL) revolves around the interactions between an agent and its environment. The environment represents the world where the agent lives and takes action. At each step, the agent observes some information about the environment, makes decisions, and affects the environment through its actions.

The agent also receives rewards from the environment, which indicate how well it is doing. The agent's ultimate goal is to maximize the total rewards it receives, called return. RL methods are designed to help the agent learn and improve its behaviors to achieve its objectives.

To better understand RL, we need to explore some key terms such as states and observations, action spaces, policies, trajectories, different ways of measuring return, the optimization problem in RL, and value functions. By understanding these fundamental concepts, we build a strong foundation to understand how RL methods work and how they can be used to solve many different real-life problems.

## States and Observations

In RL, a **state** represents a complete description of the world at a given point in time. It contains all the information necessary to understand the current state of the environment, including any hidden information. On the other hand, an **observation** provides a partial description of the state and may omit certain details. In deep RL, observations are often represented as real-valued vectors, matrices, or higher-order tensors. For example, a visual observation can be represented by an RGB matrix of pixel values, while the state of a robot can be represented by its joint angles and velocities.

## Action Spaces

Different environments allow different kinds of actions to be taken by the agent. The set of valid actions is referred to as the **action space**. In RL, there are two main types of action spaces: discrete and continuous. 

In **discrete action spaces**, such as Atari games or the game of chess, only a finite number of moves are available to the agent. The agent selects an action from a predefined set of discrete choices. 

In **continuous action spaces**, such as controlling a robot in the physical world, actions are represented as real-valued vectors. This allows for a wide range of precise actions to be taken by the agent.

## Policies

A **policy** is a rule or strategy used by an agent to decide what actions to take in a given state. In RL, policies can be deterministic or stochastic.

A **deterministic policy** selects a single action for a given state:

$$a_t = \mu (s_t)$$

The action is directly determined by the policy's output, which can be a computable function based on a set of parameters, such as the weights and biases of a neural network.

A **stochastic policy** selects actions probabilistically: 

$$a_t \sim \pi( \cdot \mid s_t)$$

The policy outputs a probability distribution over actions for a given state, and the agent samples an action based on this distribution. Stochastic policies are often used when exploring different actions or dealing with uncertainty.

In deep RL, parameterized policies are commonly used, where the outputs are computable functions that depend on a set of parameters. These policies can be represented by neural networks, allowing for flexible and expressive policy representations.

## Trajectories

A trajectory, also known as an episode or rollout, is a sequence of states and actions experienced by an agent in the environment. It captures the agent's interactions with the environment over a specific period. A trajectory can be represented as follows:

$$\tau = (s_0, a_0, s_1, a_1, ...)$$

The very first state of the trajectory, **`s₀`**, is randomly sampled from the start-state distribution, denoted as **`S₀ ~ ρ₀(·)`**. The state transitions in an environment can be either deterministic or stochastic. 

In deterministic state transitions, the next state, **`sₜ₊₁`**, is solely determined by the current state and action:

$$s_{t+1} = f(s_t, a_t)$$

In stochastic state transitions, the next state, **`sₜ₊₁`**, is sampled from a transition probability distribution:

$$s_{t+1} \sim P(\cdot|s_t, a_t)$$

## Rewards and Return

The reward function plays a crucial role in RL. It quantifies the immediate desirability or quality of a particular state-action-state transition. The reward function depends on the current state, the action taken, and the next state:

The agent's goal is to maximize the cumulative reward over a trajectory, denoted as **`R(τ)`**. There are different types of returns in RL:

**Finite-horizon undiscounted return** represents the sum of rewards obtained within a fixed window of steps:

$$R(\tau) = \sum^T_{t=0}r_t$$

**Infinite-horizon discounted return** represents the sum of all rewards obtained by the agent, but discounted based on how far they are obtained in the future:

$$R(\tau) = \sum^\infty_{t=0} \gamma^tr_t$$

where $\gamma \in (0, 1)$

## The RL Problem

The primary objective in RL is to find a policy that maximizes the expected return when the agent acts according to it. Suppose the environment transitions and the policy are stochastic. In that case, the probability of a T-step trajectory can be expressed as:

$$
P(\tau | \pi) = \rho_0(s_0) \prod^{T - 1}\_{t=0} P(s_{t+1}|s_t, a_t) \pi (a_t|s_t)
$$

The expected return, or objective function, can be defined as:

$$
J(\pi) = \int_\tau P(\tau | \pi)R(\tau) = E_{\tau \sim \pi}[R(\tau)]
$$

The central optimization problem in RL is to find the optimal policy, denoted as **`π*`**:

$$
\pi^*=\arg \max_\pi J(\pi)
$$

## Value Functions

Value functions provide estimates of the expected return associated with states or state-action pairs. There are four main types of value functions:

The **on-policy value function, `Vπ(s)`,** estimates the expected return if we start in state *s* and always act according to policy *π*:

$$V^\pi(s) = E_{\tau \sim \pi} [R(\tau)|s_0 = s]$$ 

The **on-policy action-value function** **`Qπ(s, a)`** estimates the expected return if we start in state *s*, take action *a*, and then forever act according to policy *π*:

$$Q^\pi(s,a) = E_{\tau \sim \pi}[R(\tau)|s_0=s, a_0=a]$$

The **optimal value function** **`V*(s)`** estimates the expected return if we start in state *s* and always act according to the *optimal* policy:

$$V^\pi(s) = \max_\pi E_{\tau \sim \pi} [R(\tau)|s_0 = s]$$

The **optimal action-value function** **`Q*(s, a)`** estimates the expected return if we start in state *s*, take action *a*, and then forever act according to the *optimal* policy:

$$Q^\pi(s,a) = \max_\pi E_{\tau \sim \pi}[R(\tau)|s_0=s, a_0=a]$$

## Bellman Equations

All four value functions above satisfy self-consistency equations known as the Bellman equations. These equations describe the relationship between the value of a state or state-action pair and the value of subsequent states.

The Bellman equations for on-policy value functions are:

$$V^\pi(s)=E_{a \sim \pi, s' \sim P}[r(s,a)+ \gamma V^\pi(s')],$$

$$Q^\pi(s,a)=E_{s' \sim P}[r(s,a)+ \gamma E_{a' \sim \pi}[Q^\pi(s', a')]$$

The Bellman equations for optimal value functions are:

$$V^\*(s)=\max_a E_{s' \sim P} [r(s,a)+\gamma V^*(s')],$$

$$Q^\*(s,a)= E_{s' \sim P} [r(s,a)+\gamma \max_a' Q^*(s', a')]$$

The crucial difference between the Bellman equations for on-policy value functions and optimal value functions is the absence or presence of the maximization over actions. The inclusion of this maximization reflects the fact that the agent must select the action that leads to the highest value in order to act optimally.

## Advantage Functions

In RL, there are situations where we are interested in understanding not just how good an action is in absolute terms but how much better it is compared to other available actions on average. The **advantage function** captures the relative advantage of taking a specified action in a state compared to randomly selecting an action. It can be defined as:

$$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$$

The advantage function provides insights into the superiority of a specific action in a given state, considering the current policy's performance.

## Reference:

[1] OpenAI Spinning Up. https://spinningup.openai.com/en/latest/spinningup/rl_intro.html