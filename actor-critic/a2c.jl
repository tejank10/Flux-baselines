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
γ = 0.99    # discount rate

η = 2.5e-4   # Learning rate
ρ = 0.95    # Gradient momentum for RMSProp

# Exploration params
ϵ_START = 1.0   # Initial exploration rate
ϵ_STOP = 0.1    # Final exploratin rate
ϵ_STEPS = 1000000   # Final exploration frame, using linear annealing

cᵥ = 0.5			# v loss coefficient
cₑ = 0.01 # entropy coefficient

memory = []

frames = 0

# ----------------------------- Model Architecture -----------------------------

base = Chain(Dense(4, 24, relu), Dense(24, 24, relu))
value = Dense(24, 1)
policy = Dense(24, ACTION_SIZE)

# ------------------------------------ Loss ------------------------------------

# Policy Loss
function loss_π(π, v, action, rₜ)
    logπ = log.(mean(π .* action, 1) + 1e-10)
    advantage = rₜ - v
    -logπ .* advantage.data #to stop backpropagation through advantage
end

# Value loss
lossᵥ(v, rₜ) = cᵥ * (rₜ - v) .^ 2

entropy(π) = cₑ * mean(π .* log.(π + 1e-10), 1)

# Total Loss = Policy loss + Value Loss + Entropy
function loss(x)
  s = hcat(x[1, :]...)
  a = Flux.onehotbatch(x[2, :], 1:ACTION_SIZE)
  r = hcat(x[3, :]...)
  s′ = hcat(x[4, :]...)
  s_mask = .!hcat(x[5, :]...)

  base_out = base(s)
  v = value(base_out)
  π = softmax(policy(base_out))

  v′ = value(base(s′))
  rₜ = r + γ .* v′ .* s_mask	# set v to 0 where s_ is terminal state

  mean(loss_π(π, v, a, rₜ) + lossᵥ(v, rₜ) + entropy(π))
end

# --------------------------- Training ----------------------------------------

opt = RMSProp(params(base) ∪ params(value) ∪ params(policy), η; ρ = ρ)

function train()
	x = hcat(memory...)
  Flux.back!(loss(x))
  opt()
end

# --------------------------- Helper Functions --------------------------------

get_ϵ() = frames >= ϵ_STEPS ? ϵ_STOP : ϵ_START + frames * (ϵ_STOP - ϵ_START) / ϵ_STEPS

# stores the tuple of state, action, reward, next_state, and done
function remember(state, action, reward, next_state, done)
  push!(memory, [state, action, reward, next_state, done])
end

# Choose action according to policy CartPolePolicy
function action(π::CartPolePolicy, reward, state, action)
  if rand() <= get_ϵ()
    return rand(1:ACTION_SIZE) - 1
  end

  act_prob = softmax(policy(base(state))) #action probabilities
  return sample(1:ACTION_SIZE, Weights(act_prob.data)) - 1 # returns action
end


function episode!(env, π = RandomPolicy())
  global frames
  ep = Episode(env, π) # Runs an episode with policy π

  for (s, a, r, s′) in ep
    OpenAIGym.render(env)
    r = env.done ? -1 : r
    if π.train remember(s, a + 1, r, s′, env.done) end
    frames += 1
  end

  ep.total_reward
end

# ------------------------------ Training --------------------------------------

e = 1
while frames < ϵ_STEPS
  reset!(env)
  total_reward = episode!(env, CartPolePolicy())
  println("Episode: $e | Score: $total_reward")
  train()
  memory = []
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
