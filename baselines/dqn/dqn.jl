using Flux
using OpenAIGym
using BSON: @save
import Reinforce.action
import Flux.params
using Flux.Optimise: Optimiser

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
γ = 1.0f0   # discount rate

# Exploration params
ϵ = 1.0f0   # Initial exploration rate
ϵ_MIN = 0.01f0    # Final exploratin rate
ϵ_DECAY = 0.995f0

# Optimiser params
η = 0.01f0   # Learning rate

memory = [] #used to remember past results

# ------------------------------ Model Architecture ----------------------------

model = Chain(Dense(STATE_SIZE, 24, tanh), Dense(24, 48, tanh), Dense(48, ACTION_SIZE)) |> gpu

loss(x, y) = Flux.mse(model(x), y)

opt = Optimiser(ADAM(η), InvDecay(0.01f0))

# ----------------------------- Helper Functions -------------------------------

get_ϵ(e) = max(ϵ_MIN, min(ϵ, 1.0f0 - log10(e * ϵ_DECAY)))

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
  global ϵ
  batch_size = min(BATCH_SIZE, length(memory))
  minibatch = sample(memory, batch_size, replace = false)

  x = Matrix{Float32}(undef, STATE_SIZE, batch_size)
  y = Matrix{Float32}(undef, ACTION_SIZE, batch_size)
  for (iter, (state, action, reward, next_state, done)) in enumerate(minibatch)
    target = reward
    if !done
      target += γ * maximum(model(next_state |> gpu))
    end

    target_f = model(state |> gpu)
    target_f[action] = target

    x[:, iter] .= state
    y[:, iter] .= target_f
  end
  x = x |> gpu
  y = y |> gpu

  Flux.train!(loss, params(model), [(x, y)], opt)

  ϵ *= ϵ > ϵ_MIN ? ϵ_DECAY : 1.0f0
end

function episode!(env, π = RandomPolicy())
  ep = Episode(env, π)

  for (s, a, r, s′) in ep
    #OpenAIGym.render(env)
    if π.train
      remember(s, a + 1, r, s′, env.done)
    end
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
  print("Episode: $e | Score: $total_reward ")
  if e > 100
    last_100_mean = mean(scores[(end - 99):end])
    print("Last 100 episodes mean score: $last_100_mean")
    if last_100_mean > 195
      println("\nCartPole-v0 solved!")
      break
    end
  end
  println()
  replay()
  global e += 1
end
#---------------------------------- Saving -------------------------------------
weights = Tracker.data.(params(model))
@save "dqn-weights.bson" weights
# -------------------------------- Testing -------------------------------------
global ee = 1

while true
  reset!(env)
  total_reward = episode!(env, CartPolePolicy(false))
  println("Episode: $ee | Score: $total_reward")
  global ee += 1
end
