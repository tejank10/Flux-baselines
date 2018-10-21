using Flux, CuArrays
using OpenAIGym
import Reinforce:action
import Flux.params
using CUDAnative: log

#Define custom policy for choosing action
struct CartPolePolicy <: Reinforce.AbstractPolicy
  train::Bool
  act_prob::AbstractArray
end

CartPolePolicy(train = true) = CartPolePolicy(train,
                                    ones(Float32, ACTION_SIZE)/ACTION_SIZE)
#Load game environment
env = GymEnv("CartPole-v0")

# ----------------------------- Parameters -------------------------------------

STATE_SIZE = length(env.state)
ACTION_SIZE = length(env.actions)
γ = 0.99f0    # discount rate

η = 2.5f-4   # Learning rate
ρ = 0.95f0    # Gradient momentum for RMSProp

cᵥ = 0.5f0			# v loss coefficient
cₑ = 0.01f0 # entropy coefficient

memory = []

frames = 0

# ----------------------------- Model Architecture -----------------------------

base = Chain(Dense(4, 24, relu), Dense(24, 24, relu)) |> gpu
value = Dense(24, 1) |> gpu
policy = Dense(24, ACTION_SIZE) |> gpu

# ------------------------------------ Loss ------------------------------------

# Policy Loss
function loss_π(π, v, action, rₜ)
    logπ = log.(mean(π .* action, 1) + 1f-10)
    advantage = rₜ - v
    -logπ .* advantage.data #to stop backpropagation through advantage
end

# Value loss
lossᵥ(v, rₜ) = cᵥ * (rₜ - v) .^ 2

entropy(π) = cₑ * mean(π .* log.(π + 1f-10), 1)

# Total Loss = Policy loss + Value Loss + Entropy
function loss(x)
  s = hcat(x[1, :]...) |> gpu
  a = hcat(x[2, :]...) |> gpu
  r = hcat(x[3, :]...) |> gpu
  s′ = hcat(x[4, :]...) |> gpu
  s_mask = .!hcat(x[5, :]...) |> gpu

  base_out = base(s)
  v = value(base_out)
  π = softmax(policy(base_out))
  v′ = value(base(s′))
  rₜ = r + γ * v′ .* s_mask	# set v to 0 where s_ is terminal state

  mean(loss_π(π, v, a, rₜ) + lossᵥ(v, rₜ) + entropy(π))
end

# --------------------------- Training ----------------------------------------

opt = RMSProp(vcat(params.((base, value, policy)...)), η; ρ = ρ)

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
  act_prob = policy(state |> gpu) #action probabilities
  π.act_prob .= act_prob
  rand(Multinomial(1, act_prob.data)) - 1 # returns action
end

function MakeMemory(buffer::Dict{Symbol, Array}, ks)
  obs, a, r, a_dist = (vcat(buffer[key]) for key in ks)

end

function episode!(env, π = RandomPolicy())
  ep = Episode(env, π) # Runs an episode with policy π
  ks = [:s, :a, :r, :a_dist]
  buffer = Dict{String,Array}(key=>[] for key in ks)

  remember(key::Symbol, value) = push!(buffer, value)

  for (s, a, r, s′) in ep
    #OpenAIGym.render(env)
    π.train && remember.(keys, [s, a, r, π.act_prob])
  end

  MakeMemory(buffer)

end

function step!(numEp::UInt)
  for e=1:numEp
    env.reset!()
    episode!(env, CartPolePolicy())
  end
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
