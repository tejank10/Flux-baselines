using Flux, CuArrays
using BSON:@save
using Flux:params
using OpenAIGym
import Reinforce:action
import Base: deepcopy
ENV["CUDA_VISIBLE_DEVICES"] = 4
# ------------------------ Load game environment -------------------------------
env = GymEnv("Pong-v0")

# Custom Policy for Pong-v0
mutable struct PongPolicy <: Reinforce.AbstractPolicy
  prev_states
  prev_act
  train::Bool
  function PongPolicy(train = true)
    prev_state = preprocess(env.state)
    prev_states = cat(3, prev_state, prev_state, prev_state)
    prev_states = reshape(prev_states, size(prev_states)..., 1)
    new(prev_states, 2, train)
  end
end

# ---------------------------- Parameters --------------------------------------

STATE_SIZE = 6400 #length(env.state)
ACTION_SPACE = 3 #length(env.actions)
MEM_SIZE = 100000
BATCH_SIZE = 32
REPLAY_START_SIZE = 10000
UPDATE_FREQ = 1000
MAX_TRAIN_STEPS = 50000
γ = 0.99f0    # discount rate

# Exploration params
ϵ_START = 1.0   # Initial exploration rate
ϵ_STOP = 0.02    # Final exploratin rate
ϵ_STEPS = 100000   # Final exploration frame, using linear annealing

# Optimiser params
η = 0.0001f0   # Learning rate
ρ = 0.95f0    # Gradient momentum for RMSProp

memory = [] #used to remember past results
frames = 0
C = 0

# --------------------------- Model Architecture -------------------------------
struct nn
  base
  value
  adv
  opt

  function nn(base, value, adv)
    all_params = vcat(params(base), params(value), params(adv))
    opt = ADAM(all_params, η)
    new(base, value, adv, opt)
  end
end

function (m::nn)(inp)
  Q(x::TrackedArray) = m.value(x) .+ m.adv(x) .- mean(m.adv(x), 1)
  Q(m.base(inp))
end

function deepcopy(m::nn)
  nn(deepcopy(m.base), deepcopy(m.value), deepcopy(m.adv))
end

base = Chain(Conv((8,8), 4=>32, relu; stride=(4,4)),
             Conv((4,4), 32=>64, relu; stride=(2,2)),
             Conv((3,3), 64=>64, relu; stride=(1,1)), x->reshape(x, :, size(x, 4))) |> gpu
value = Chain(Dense(6*6*64, 512, relu), Dense(512, 1)) |> gpu
adv = Chain(Dense(6*6*64, 512, relu), Dense(512, ACTION_SPACE)) |> gpu

model = nn(base, value, adv)

model_target = deepcopy(model)

huber_loss(x, y) = mean(sqrt.(1 + (model(x) - y) .^ 2) - 1)

fit_model(data) = Flux.train!(huber_loss, data, model.opt)

# ------------------------------- Helper Functions -----------------------------

get_ϵ() = frames > ϵ_STEPS ? ϵ_STOP : ϵ_START + frames * (ϵ_STOP - ϵ_START) / ϵ_STEPS

function save_model(model::nn)
  base_wt = cpu.(Tracker.data.(params(model.base)))
  val_wt = cpu.(Tracker.data.(params(model.value)))
  adv_wt = cpu.(Tracker.data.(params(model.adv)))

  @save "../models/duel_dqn_base" base_wt
  @save "../models/duel_dqn_val" val_wt
  @save "../models/duel_dqn_adv" adv_wt

  println("Model saved")
end

function preprocess(I)
  #= preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector =#
  I = I[36:195, :, :] # crop
  I = I[1:2:end, 1:2:end, 1] # downsample by factor of 2
  I[I .== 144] = 0 # erase background (background type 1)
  I[I .== 109] = 0 # erase background (background type 2)
  I[I .!= 0] = 1 # everything else (paddles, ball) just set to 1
  I = Float32.(reshape(I, 80,80,1))
  return I#[:] #Flatten and return
end

# Putting data into replay buffer
function remember(prev_s, s, a, r, s′, done)
  if length(memory) == MEM_SIZE
    deleteat!(memory, 1)
  end
  r = Float32(r)
  state = cat(3, preprocess(s), prev_s)

  next_state = state[:,:,1:3,1]
  next_state = cat(3, preprocess(s′), next_state)
  next_state = reshape(next_state, size(next_state)..., 1)
  push!(memory, (state, a, r, next_state, done))
end

function action(π::PongPolicy, reward, state, action)
  #if frames % 4 != 0 return π.prev_act end
  if rand() <= get_ϵ() && π.train
    π.prev_act = rand(1:ACTION_SPACE) 
    return π.prev_act # UP and DOWN action corresponds to 2 and 3
  end

  s = cat(3, preprocess(state), π.prev_states) |> gpu
  act_values = model(s)
  π.prev_act = Flux.argmax(act_values)[1]
  
  return π.prev_act  # returns action max Q-value
end

function replay()
  global C, model_target

  minibatch = sample(memory, BATCH_SIZE, replace = false)
  x = zeros(80, 80, 4, BATCH_SIZE)
  y = zeros(ACTION_SPACE, BATCH_SIZE)
  i = 1
  for (state, action, reward, next_state, done) in minibatch
    target = reward
    if !done
      a_max = Flux.argmax(model(next_state |> gpu))[1]
      target += γ * model_target(next_state |> gpu).data[a_max]
    end
    target_f = model(state |> gpu).data
    target_f[action] = target

    x[:, :, :, i] .= state[:, :, :, 1]
    y[:, i] .= target_f[:, 1]

    #dataset = zip(cu(state), cu(target_f))
    #fit_model(dataset)

    C = (C + 1) % UPDATE_FREQ
    i += 1
  end
  loss = huber_loss(cu(x), cu(y))
  Flux.back!(loss)
  model.opt()
  # Update target model
  if steps % UPDATE_FREQ == 0
    model_target = deepcopy(model)
    save_model(model)
  end

  return loss.data
end

function episode!(env, π = RandomPolicy())
  global frames, steps
  ep = Episode(env, π)
  for (s, a, r, s′) in ep
    #OpenAIGym.render(env)
    if π.train remember(π.prev_states, s, a, r, s′, r == -1) end
    π.prev_states = cat(3, preprocess(s), π.prev_states[:,:,1:2,1])
    π.prev_states = reshape(π.prev_states, size(π.prev_states)..., 1)
    frames += 1
    if frames >= REPLAY_START_SIZE
      replay()
      steps += 1
    end
    
  end
  ep.total_reward
end

# ------------------------------ Training --------------------------------------

e = 1
steps = 0
scores = zeros(100)
idx = 1
while true
  reset!(env)
  total_reward = episode!(env, PongPolicy())
  scores[idx] = total_reward
  idx = idx % 100 + 1
  eps = get_ϵ()
  avg_score = mean(scores)
  println("Episode: $e | Score: $total_reward | eps: $eps | steps: $steps | Avg Score: $avg_score")
  e += 1
end
#=
# -------------------------------- Testing -------------------------------------
ee = 1

while true
  reset!(env)
  total_reward = episode!(env, PongPolicy(false))
  println("Episode: $ee | Score: $total_reward")
  ee += 1
end=#
