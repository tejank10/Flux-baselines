include("utils/memory.jl")

using Flux
using Flux:params
using OpenAIGym
import Reinforce:action
import Flux.params

# Load game environment
env = GymEnv("CartPole-v0")

#Define custom policy for choosing action
mutable struct CartPolePolicy <: Reinforce.AbstractPolicy
  train::Bool

  function CartPolePolicy(train = true)
    new(train)
  end
end

# ------------------------------ Parameters ------------------------------------

STATE_SIZE = length(env.state)
ACTION_SIZE = length(env.actions)
MEM_SIZE = 1000000
BATCH_SIZE = 32
UPDATE_FREQ = 10000
γ = 0.99    # discount rate

# Exploration params
ϵ_START = 1.0   # Initial exploration rate
ϵ_STOP = 0.1    # Final exploratin rate
ϵ_STEPS = 1000000   # Final exploration frame, using linear annealing

# Optimiser params
η = 2.5e-4   # Learning rate
ρ = 0.95    # Gradient momentum for RMSProp

C = 0
frames = 0

memory = Memory(MEM_SIZE, STATE_SIZE)

# ------------------------------ Model Architecture ----------------------------
model = Chain(Dense(STATE_SIZE, 24, σ), Dense(24, 24, σ), Dense(24, ACTION_SIZE))
target_model = deepcopy(model)

#Loss
function loss(x, y, ISWeights)
  sq_diff = (x - y) .^ 2
  cost = mean(broadcast(*, sq_diff, reshape(ISWeights, 1, length(ISWeights))))
  return cost
end

#Absolute Error (Used for SumTree)
abs_errors(x, y) = sum(abs.(x - y), 1)

#Optimizer
opt = RMSProp(params(model), η; ρ = ρ)

# ------------------------------------------------------------------------------

get_ϵ() = frames >= ϵ_STEPS ? ϵ_STOP : ϵ_START + frames * (ϵ_STOP - ϵ_START) / ϵ_STEPS

function remember(state, action, reward, next_state, done)
  transition = vcat(state, [action, reward], next_state)
  store!(memory, transition)
end

function action(π::CartPolePolicy, reward, state, action)
  if rand() <= get_ϵ() && π.train
    return rand(1:ACTION_SIZE) - 1
  end

  act_values = model(state)
  return Flux.argmax(act_values) - 1
end

function replay()
  global C, model_target
  if C == 0
    model_target = deepcopy(model)
  end

  tree_idx, batch_memory, ISWeights = mem_sample(memory, BATCH_SIZE)

  states = batch_memory[1:STATE_SIZE, :]
  next_states = batch_memory[end - STATE_SIZE + 1:end, :]

  q_next, q_curr = target_model(next_states).data, model(states)

  q_target = q_curr.data
  eval_act_index = Int32.(batch_memory[STATE_SIZE + 1, :])
  reward = batch_memory[STATE_SIZE + 2, :]

  for i = 1:BATCH_SIZE
    a_max = Flux.argmax(model(next_states))
    q_target[eval_act_index[i], i] = reward[i] + γ * maximum(q_next[:, a_max])
  end

  # Train
  cost = loss(q_curr, q_target, ISWeights)
  Flux.back!(cost)
  opt()

  C = (C + 1) % UPDATE_FREQ

  # update priority
  abs_error = abs_errors(q_curr, q_target).data
  batch_update!(memory, tree_idx, abs_error)
end

function episode!(env, π = RandomPolicy())
  global frames
  ep = Episode(env, π)

  for (s, a, r, s′) in ep
    OpenAIGym.render(env)
    r = env.done ? -1 : r
    if π.train remember(s, a + 1, r, s′, env.done) end
    frames += 1
    replay()
  end

  ep.total_reward
end

# ------------------------------ Training --------------------------------------

e = 1
while frames < ϵ_STEPS
  reset!(env)
  total_reward = episode!(env, CartPolePolicy())
  println("Episode: $e | Score: $total_reward")
  e += 1
end

# -------------------------------- Testing -------------------------------------
ee = 1

while true
  reset!(env)
  total_reward = episode!(env, CartPolePolicy(false))
  println("Episode: $ee | Score: $total_reward")
  ee += 1
end
