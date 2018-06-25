using Flux, CuArrays
using Flux:params
using OpenAIGym
import Reinforce:action

# ------------------------ Load game environment -------------------------------
env = GymEnv("Pong-v0")

# Custom Policy for Pong-v0
mutable struct PongPolicy <: Reinforce.AbstractPolicy
  prev_state
  train::Bool
  function PongPolicy(train = true)
    new(zeros(STATE_SIZE), train)
  end
end

# ---------------------------- Parameters --------------------------------------

STATE_SIZE = 6400 #length(env.state)
ACTION_SPACE = 2 #length(env.actions)
MEM_SIZE = 1000000
BATCH_SIZE = 32
REPLAY_START_SIZE = 5000
UPDATE_FREQ = 500
γ = 0.99    # discount rate

# Exploration params
ϵ_START = 1.0   # Initial exploration rate
ϵ_STOP = 0.1    # Final exploratin rate
ϵ_STEPS = 1000000   # Final exploration frame, using linear annealing

# Optimiser params
η = 2.5e-4   # Learning rate
ρ = 0.95    # Gradient momentum for RMSProp

memory = [] #used to remember past results
frames = 1
C = 0

# --------------------------- Model Architecture -------------------------------

value = Dense(200, 1, relu) |> gpu
adv = Dense(200, ACTION_SPACE, relu) |> gpu

Q(x::TrackedArray) = value(x) .+ adv(x) .- mean(adv(x), 1)

model = Chain(Dense(STATE_SIZE, 200, relu), x -> Q(x)) |> gpu
model_target = deepcopy(model)

huber_loss(x, y) = mean(sqrt.(1 + (model(x) - y) .^ 2) - 1)

opt() = RMSProp(params(model), η; ρ = ρ)

fit_model(data) = Flux.train!(huber_loss, data, opt)

# ------------------------------- Helper Functions -----------------------------

get_ϵ() = frames == ϵ_STEPS ? ϵ_STOP : ϵ_START + frames * (ϵ_STOP - ϵ_START) / ϵ_STEPS

function preprocess(I)
  #= preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector =#
  I = I[36:195, :, :] # crop
  I = I[1:2:end, 1:2:end, 1] # downsample by factor of 2
  I[I .== 144] = 0 # erase background (background type 1)
  I[I .== 109] = 0 # erase background (background type 2)
  I[I .!= 0] = 1 # everything else (paddles, ball) just set to 1

  return I[:] #Flatten and return
end

# Putting data into replay buffer
function remember(prev_s, s, a, r, s′, done)
  if length(memory) == MEM_SIZE
    deleteat!(memory, 1)
  end

  state = preprocess(s) - prev_s |> gpu
  next_state = env.done ? zeros(STATE_SIZE) : preprocess(s′)
  next_state = next_state - preprocess(s) |> gpu
  push!(memory, (state, a, r, next_state, done))
end

function action(π::PongPolicy, reward, state, action)
  if rand() <= get_ϵ() && π.train
    return rand(1:ACTION_SPACE) + 1 # UP and DOWN action corresponds to 2 and 3
  end

  s = preprocess(state) - π.prev_state |> gpu
  act_values = model(s)
  return Flux.argmax(act_values) + 1  # returns action max Q-value
end

function replay()
  global C, model_target

  minibatch = sample(memory, BATCH_SIZE, replace = false)

  for (state, action, reward, next_state, done) in minibatch
    target = reward

    if !done
      a_max = Flux.argmax(model(next_state))
      target += γ * model_target(next_state).data[a_max]
    end

    target_f = model(state).data
    target_f[action] = target
    dataset = zip(state, cu(target_f))
    fit_model(dataset)

    # Update target model
    if C == 0
      model_target = deepcopy(model)
    end

    C = (C + 1) % UPDATE_FREQ
  end
end

function episode!(env, π = RandomPolicy())
  global frames
  ep = Episode(env, π)

  for (s, a, r, s′) in ep
    #OpenAIGym.render(env)
    if π.train remember(π.prev_state, s, a - 1, r, s′, env.done) end
    π.prev_state = preprocess(s)
    frames += 1
  end

  ep.total_reward
end

# ------------------------------ Training --------------------------------------

e = 1
while frames < ϵ_STEPS
  reset!(env)
  total_reward = episode!(env, PongPolicy())
  println("Episode: $e | Score: $total_reward")
  if frames >= REPLAY_START_SIZE
    replay()
  end
  e += 1
end

# -------------------------------- Testing -------------------------------------
ee = 1

while true
  reset!(env)
  total_reward = episode!(env, PongPolicy(false))
  println("Episode: $ee | Score: $total_reward")
  ee += 1
end
