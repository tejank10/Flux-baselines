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
MEM_SIZE = 100000
BATCH_SIZE = 64
UPDATE_FREQ = 10000
γ = 1.0f0   # discount rate

# Exploration params
ϵ = 1.0f0   # Initial exploration rate
ϵ_STOP = 0.01f0    # Final exploratin rate
ϵ_DECAY = 0.995f0

# Optimiser params
η = 0.01f0   # Learning rate

memory = [] #used to remember past results

# ------------------------------ Model Architecture ----------------------------

model = Chain(Dense(STATE_SIZE, 24, tanh), Dense(24, 48, tanh), Dense(48, ACTION_SIZE)) |> gpu
model_target = deepcopy(model)

loss(x, y) = Flux.mse(model(x), y)

opt = ADAM(params(model), η; decay = 0.01f0)

fit_model(dataset) = Flux.train!(loss, dataset, opt)

# ----------------------------- Helper Functions -------------------------------

get_ϵ(e) = max(ϵ_STOP, min(ϵ, 1.0f0-log10(e * ϵ_DECAY)))

function remember(state, action, reward, next_state, done)
  if length(memory) == MEM_SIZE
    deleteat!(memory, 1)
  end
  push!(memory, (state, action, reward, next_state, done))
end

function action(π::CartPolePolicy, reward, state, action)
  if rand() <= get_ϵ(e) && π.train
    return rand(1:ACTION_SIZE) - 1
  end

  act_values = model(state |> gpu)
  return Flux.argmax(act_values) - 1
end

function replay()
  global ϵ, ϵ_STOP, ϵ_DECAY
  minibatch = sample(memory, BATCH_SIZE, replace = false)
  x = Matrix{Float32}(STATE_SIZE, BATCH_SIZE)
  y = Matrix{Float32}(ACTION_SIZE, BATCH_SIZE)
  for (iter, (state, action, reward, next_state, done)) in enumerate(minibatch)
    target = reward
    if !done
      target += γ * maximum(model(next_state |> gpu).data)
    end

    target_f = model(state |> gpu).data
    target_f[action] = target
    
    x[:, iter] .= state
    y[:, iter] .= target_f
  end
  x = x |> gpu
  y = y |> gpu
  fit_model([(x, y)])
  if ϵ > ϵ_STOP
    ϵ *= ϵ_DECAY
  end
end

function episode!(env, π = RandomPolicy())
  ep = Episode(env, π)

  for (s, a, r, s′) in ep
    #OpenAIGym.render(env)
    if π.train remember(s, a + 1, r, s′, env.done) end
  end

  ep.total_reward
end

# ------------------------------ Training --------------------------------------

e = 1
while e < 1000
  reset!(env)
  total_reward = episode!(env, CartPolePolicy())
  println("Episode: $e | Score: $total_reward")
  if length(memory) > BATCH_SIZE
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
