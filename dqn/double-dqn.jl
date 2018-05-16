using Flux
using OpenAIGym
import Reinforce.action
import Flux.params

#Define custom policy for choosing action
mutable struct CartPolePolicy <: Reinforce.AbstractPolicy
  train::Bool

  function CartPolePolicy(train = true)
    new(train)
  end
end

#Load game environment
env = GymEnv("CartPole-v0")

# ----------------------------- Parameters -------------------------------------

STATE_SIZE = length(env.state)
ACTION_SIZE = length(env.actions)
MEM_SIZE = 1000000
BATCH_SIZE = 32
REPLAY_START_SIZE = 50000
UPDATE_FREQ = 10000
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

# ------------------------------ Model Architecture ----------------------------

model = Chain(Dense(STATE_SIZE, 24, σ), Dense(24, 24, σ), Dense(24, ACTION_SIZE))
model_target = deepcopy(model)

huber_loss(x, y) = mean(sqrt.(1 + (model(x) - y) .^ 2) - 1)

opt = RMSProp(params(model), η; ρ = ρ)

fit_model(dataset) = Flux.train!(loss, dataset, opt)

# ----------------------------- Helper Functions -------------------------------

get_ϵ() = frames >= ϵ_STEPS ? ϵ_STOP : ϵ_START + frames * (ϵ_STOP - ϵ_START) / ϵ_STEPS

function remember(state, action, reward, next_state, done)
  if length(memory) == MEM_SIZE
    deleteat!(memory, 1)
  end
  push!(memory, (state, action, reward, next_state, done))
end

function action(π::CartPolePolicy, reward, state, action)
  if rand() <= get_ϵ() && π.train
    return rand(1:ACTION_SIZE) - 1
  end

  act_values = model(state)
  return Flux.argmax(act_values) - 1
end

function replay()
  global model_target, C
  minibatch = sample(memory, BATCH_SIZE, replace = false)

  for (state, action, reward, next_state, done) in minibatch
    target = reward

    if !done
      a_max = Flux.argmax(model(next_state))
      target += γ * model_target(next_state).data[a_max]
    end

    target_f = model(state).data
    target_f[action] = target
    dataset = zip(state, target_f)
    fit_model(dataset)

    if C == 0
      model_target = deepcopy(model)
    end

    C = (C + 1) % UPDATE_FREQ
  end
end

function episode!(env, π = RandomPolicy())
  ep = Episode(env, π)

  for (s, a, r, s′) in ep
    OpenAIGym.render(env)
    r = env.done ? -1 : r
    if π.train remember(s, a + 1, r, s′, env.done) end
  end

  ep.total_reward
end

# ------------------------------ Training --------------------------------------

e = 1
while frames < ϵ_STEPS
  reset!(env)
  total_reward = episode!(env, CartPolePolicy())
  println("Episode: $e | Score: $total_reward")
  if length(memory) >= REPLAY_START_SIZE
    replay()
  end
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
