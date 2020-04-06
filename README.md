# Flux Baselines
Implementations of reinforcement learning algorithms using Flux
NOTE: The code works on Julia 0.6.x. Upgrade to 1.0 will be released soon.


## Updated to 1.x

Actor-critic
- [ ] A2C

DDPG
- [ ] DDPG

DQN
- [x] Double DQN
- [x] DQN
- [ ] Duel DQN
- [x] Prioritized Replay DQN


## Dependencies
- [Flux.jl](https://www.github.com/FluxML/Flux.jl)
- [OpenAIGym.jl](https://github.com/JuliaML/OpenAIGym.jl)

## Implemented Algorithms
- Deep Q Networks [[paper](https://arxiv.org/abs/1312.5602)]
- Double DQN [[paper](https://arxiv.org/abs/1511.06581)]
- Dueling Double DQN [[paper](https://arxiv.org/abs/1511.06581)]
- Prioritized Experience DQN [[paper](https://arxiv.org/abs/1511.05952)]
- A2C [[paper](https://arxiv.org/abs/1602.01783)]
- Deep Deterministic Policy Gradients [[paper](https://arxiv.org/pdf/1509.02971.pdf)]
