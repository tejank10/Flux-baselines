include("utils/memory.jl")

using Flux, Images
using Flux:params
using OpenAIGym
import Reinforce:action
import Flux.params
using CuArrays
# -------------------------------- Constants -----------------------------------
IMG_H = 84
IMG_W = 84
IMG_C = 2

# Load game environment
env = GymEnv("Seaquest-v0")

#Define custom policy for choosing action
mutable struct SeaquestPolicy <: Reinforce.AbstractPolicy
  prev_state
  train::Bool

  function SeaquestPolicy(train = true)
    prev_state = preprocess(env.state)
    prev_state = reshape(prev_state, size(prev_state)..., 1)
    new(prev_state, train)
  end
end

# ------------------------------- Utilities ------------------------------------

function preprocess(img)
  rgb = imresize(img, IMG_W, IMG_H)

  r, g, b = rgb[:,:,1], rgb[:,:,2], rgb[:,:,3]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b     # extract luminance

  o = Float32.(gray) / 128 - 1    # normalize
end

# ------------------------------ Parameters ------------------------------------

STATE_SIZE = IMG_W * IMG_H * IMG_C
ACTION_SIZE = length(env.actions)
MEM_SIZE = 10
BATCH_SIZE = 32
UPDATE_FREQ = 10000
γ = 0.99f0    # discount rate

EMPTY_STATE = zeros(Float32, IMG_W, IMG_H, IMG_C)

# Exploration params
ϵ_MAX = 1.0   # Initial exploration rate
ϵ_MIN = 0.01    # Final exploratin rate
MAX_ϵ_STEPS = 500000   # Final exploration frame, using linear annealing
λ = -log(0.01) / MAX_ϵ_STEPS

# Optimiser params
η = 0.00025f0   # Learning rate
HUBER_LOSS_δ = 2.0f0

C = 0
memory = Memory(MEM_SIZE, STATE_SIZE)
frames = 0

# ------------------------------ Model Architecture ----------------------------
model = Chain(Conv((8,8), IMG_C=>32, relu; stride=(4,4)),
              Conv((4,4), 32=>64, relu; stride=(2,2)),
              Conv((3,3), 64=>64, relu),
              x->reshape(x, 7 * 7 * 64, :),
              Dense(7 * 7 * 64, 512, relu), Dense(512, ACTION_SIZE)) |> gpu
target_model = deepcopy(model)

# Huber Loss
function huber_loss(ŷ, y, ISWeights)
  err = y .- ŷ
  cond_ = abs.(err) .< HUBER_LOSS_δ

  L2 = 0.5f0 * err .^ 2
  L1 = HUBER_LOSS_δ * (abs.(err) - 0.5f0 * HUBER_LOSS_δ)
  loss = cond_ .* L2 
  aa = .!cond_
  bb=aa.*L1
  loss += bb
  #ISW = reshape(ISWeights, 1, length(ISWeights))
  mean(loss)# .* ISW)
end

#Absolute Error (Used for SumTree)
abs_errors(x, y) = sum(abs.(x - y), 1)

#Optimizer
opt = RMSProp(params(model), η)

# ------------------------------------------------------------------------------

get_ϵ() = ϵ_MIN + (ϵ_MAX - ϵ_MIN) * exp(-λ * frames)

function remember(prev_s, state, action, reward, next_state, done)
  global frames
  state = cat(3, preprocess(state), prev_s)

  state_ = state[:,:,1:1,1]
  if done next_state = zeros(next_state) end
  state_ = cat(3, preprocess(next_state), state_)
  state_ = reshape(state_, size(state_)..., 1)

  transition = vcat(vec(state), [action, reward], vec(state_))

  store!(memory, transition)
  frames += 1
end

function action(π::SeaquestPolicy, reward, state, action)
  if rand() <= get_ϵ() && π.train
    return rand(1:ACTION_SIZE) - 1
  end

  act_values = model(state |> gpu)
  return Flux.argmax(act_values) - 1
end

function replay()
  global C, target_model
  if C == 0
    target_model = deepcopy(model)
  end

  tree_idx, batch_memory, ISWeights = mem_sample(memory, BATCH_SIZE)

  states = batch_memory[1:STATE_SIZE, :]
  states = reshape(states, IMG_W, IMG_H, IMG_C, :)

  next_states = batch_memory[end - STATE_SIZE + 1:end, :]
  next_states = reshape(next_states, IMG_W, IMG_H, IMG_C, :)

  q_target_next, q_next = target_model(next_states |> gpu), model(next_states |> gpu)
                 q_curr = model(states |> gpu)

  q_target = q_curr.data
  eval_act_index = Int32.(batch_memory[STATE_SIZE + 1, :])
  reward = batch_memory[STATE_SIZE + 2, :]

  for i = 1:BATCH_SIZE
    q_target[eval_act_index[i], i] = reward[i]
    next_state = next_states[:,:,:,i]
    if !all(next_state .== EMPTY_STATE)
      a_max = Flux.argmax(q_next[:, i])
      q_target[eval_act_index[i], i] += γ * q_target_next[:, i].data[a_max]
    end
  end

  # Train
  cost = huber_loss(q_curr, q_target, ISWeights)
  Flux.back!(cost)
  opt()

  C = (C + 1) % UPDATE_FREQ

  # update priority
  abs_error = abs_errors(q_curr, q_target).data
  batch_update!(memory, tree_idx, abs_error)

end

function episode!(env, π = RandomPolicy())
  ep = Episode(env, π)

  for (s, a, r, s′) in ep
    #OpenAIGym.render(env)
    r = clamp(r, -1, 1)
    if π.train remember(π.prev_state, s, a + 1, r, s′, env.done) end

    if frames >= MEM_SIZE
      replay()
    end
    
    π.prev_state = preprocess(s)
    π.prev_state = reshape(π.prev_state, size(π.prev_state)..., 1)
  end

  ep.total_reward
end

# ------------------------------ Training --------------------------------------

e = 1
scores = []
while true
  reset!(env)
  total_reward = episode!(env, SeaquestPolicy())
  push!(scores, total_reward)
  ϵ = get_ϵ()
  print("Episode: $e | Score: $total_reward | $ϵ | ")
  if e > 100
    last_100_mean = mean(scores[end-99:end])
    print("Last 100 scores mean: $last_100_mean")
    if last_100_mean >= 500
      break
    end
  end
  println()
  e += 1
end

# -------------------------------- Testing -------------------------------------
ee = 1

while true
  reset!(env)
  print("1")
  total_reward = episode!(env, SeaquestPolicy(false))
  println("Episode: $ee | Score: $total_reward")
  ee += 1
end
