include("../../utils/memory.jl")

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
MEM_SIZE = 10000
BATCH_SIZE = 32
UPDATE_FREQ = 500
γ = 0.99f0    # discount rate

# Exploration params
ϵ = 1.0f0   # Initial exploration rate
ϵ_MIN = 0.01f0    # Final exploratin rate
ϵ_DECAY = 0.995f0   # Final exploration frame, using linear annealing

# Optimiser params
η = 0.01f0   # Learning rate

C = 0
memory = Memory(MEM_SIZE, STATE_SIZE)
frames = 0
# ------------------------------ Model Architecture ----------------------------
model = Chain(Dense(STATE_SIZE, 24, tanh), Dense(24, 48, tanh), Dense(48, ACTION_SIZE)) |> gpu
target_model = deepcopy(model)

#Loss
function loss(x, y, ISWeights)
  x, y, ISWeights
  sq_diff = (x .- y) .^ 2
  reshaped_ISWeights = reshape(ISWeights, 1, length(ISWeights)) |> gpu
  cost = mean(sq_diff .* reshaped_ISWeights)
  return cost
end

#Absolute Error (Used for SumTree)
abs_errors(x, y) = sum(abs.(x - y), dims=1)

#Optimizer
opt = ADAM(η)

# ------------------------------------------------------------------------------

function remember(state, action, reward, next_state, done)
  global frames
  transition = vcat(state, [action, reward], next_state)
  store!(memory, transition)
  frames += 1
end

function action(π::CartPolePolicy, reward, state, action)
  if rand() <= ϵ && π.train
    return rand(1:ACTION_SIZE) - 1
  end

  act_values = model(state |> gpu)
  return Flux.argmax(act_values) - 1
end

function replay()
  global C, target_model, ϵ
  if C == 0
    target_model = deepcopy(model)
  end

  tree_idx, batch_memory, ISWeights = mem_sample(memory, BATCH_SIZE)

  states = batch_memory[1:STATE_SIZE, :] |> gpu
  next_states = batch_memory[end - STATE_SIZE + 1:end, :] |> gpu

  q_next, q_curr = target_model(next_states), model(states)

  q_target = q_curr
  eval_act_index = Int32.(batch_memory[STATE_SIZE + 1, :])
  reward = batch_memory[STATE_SIZE + 2, :]

  for i = 1:BATCH_SIZE
    q_target[eval_act_index[i], i] = reward[i] + γ * maximum(q_next[:, i])
  end

  # Train
  Flux.train!(loss, params(model), [(q_curr, q_target, ISWeights)], opt)

  C = (C + 1) % UPDATE_FREQ

  # update priority
  abs_error = abs_errors(q_curr, q_target)
  batch_update!(memory, tree_idx, abs_error)

  ϵ *= ϵ > ϵ_MIN ? ϵ_DECAY : 1.0f0
end

function episode!(env, π = RandomPolicy())
  ep = Episode(env, π)

  for (s, a, r, s′) in ep
    #OpenAIGym.render(env)
    if π.train remember(s, a + 1, r, s′, env.done) end
  end

  if frames >= MEM_SIZE
    replay()
  end

  ep.total_reward
end

# ------------------------------ Training --------------------------------------

global e = 1
global scores = []
while true
  reset!(env)
  total_reward = episode!(env, CartPolePolicy())
  push!(scores, total_reward)
  print("Episode: $e | Score: $total_reward | $ϵ | ")
  if e > 100
    last_100_mean = mean(scores[end-99:end])
    print("Last 100 scores mean: $last_100_mean")
    if last_100_mean >= 195
      println("CartPole-v0 Solved!")
      break
    end
  end
  println()
  global e += 1
end

# -------------------------------- Testing -------------------------------------
global ee = 1

while true
  reset!(env)
  total_reward = episode!(env, CartPolePolicy(false))
  println("Episode: $ee | Score: $total_reward")
  global ee += 1
end
