using Flux, CuArrays, Flux.Tracker
using OpenAIGym
import Reinforce.action
import Flux.params
using Distributions: Uniform

#Define custom policy for choosing action
mutable struct PendulumPolicy <: Reinforce.AbstractPolicy
  train::Bool

  function PendulumPolicy(train = true)
    new(train)
  end
end

#Load game environment
env = GymEnv("Pendulum-v0")

# ----------------------------- Parameters -------------------------------------

STATE_SIZE = length(env.state)
ACTION_SIZE = length(env.actions) # Action is continuous in case of Pendulum
ACTION_BOUND = Float32(env.actions.hi[1])
BATCH_SIZE = 128
MEM_SIZE = 1000000
γ = 0.99f0     # discount rate
τ = 0.01f0 # for running average while updating target networks
η_act = 0.0001f0   # Learning rate
η_crit = 0.001f0

MAX_EP = 2000
MAX_EP_LEN = 1000
MAX_FRAMES = 12000

memory = []

frames = 0

w_init(dims...) = rand(Uniform(-0.003f0, 0.003f0), dims...)

# -------------------------------- Action Noise --------------------------------

mutable struct OUNoise
  mu
  theta
  sigma
  X
end

ou = OUNoise(0.0f0, 0.15f0, 0.2f0, [0.0f0])

function sample_noise(ou::OUNoise)
  dx = ou.theta * (ou.mu - ou.X)
  dx = dx + ou.sigma * randn(Float32, length(ou.X))
  ou.X = ou.X + dx
end
# ----------------------------- Model Architecture -----------------------------

actor = Chain(Dense(STATE_SIZE, 256, relu), Dense(256, 256, relu), 
	      Dense(256, ACTION_SIZE, tanh, initW = w_init, initb = w_init),
              x -> x * ACTION_BOUND) |> gpu
actor_target = deepcopy(actor)

critic = Chain(Dense(STATE_SIZE + ACTION_SIZE, 256, relu), Dense(256, 256, relu), 
	       Dense(256, 1, initW=w_init, initb=w_init)) |> gpu
critic_target = deepcopy(critic)
# ------------------------------- Param Update Functions---------------------------------

function update_target!(target, model; τ = 1.0f0)
  for (p_t, p_m) in zip(params(target), params(model))
    p_t.data .= (1.0f0 - τ) * p_t.data .+ τ * p_m.data
  end
end

function nullify_grad!(p)
  if typeof(p) <: TrackedArray
    p.grad .= 0.0f0
  end
  return p
end

function zero_grad!(model)
  model = mapleaves(nullify_grad!, model)
end

# ---------------------------------- Training ----------------------------------

opt_crit = ADAM(params(critic), η_crit)
opt_act = ADAM(params(actor), η_act)

function train()
  # Getting data in shape
  minibatch = sample(memory, BATCH_SIZE)
	x = hcat(minibatch...)

  s = hcat(x[1, :]...)
  a = hcat(x[2, :]...)
  r = hcat(x[3, :]...) |> gpu
  s′ = hcat(x[4, :]...) |> gpu
  s_mask = .!hcat(x[5, :]...) |> gpu

  # Update Critic
  a′ = actor_target(s′).data
  crit_tgt_in = vcat(s′, a′)
  v′ = critic_target(crit_tgt_in).data
  y = r + γ * v′ .* s_mask	# set v′ to 0 where s_ is terminal state

  crit_in = vcat(s, a) |> gpu
  v = critic(crit_in)
  loss_crit = Flux.mse(y, v)

  # Update Actor
  actions = actor(s |> gpu)
  crit_in = param(vcat(s |> gpu, actions.data))
  crit_out = critic(crit_in)
  Flux.back!(sum(crit_out))
  #grads = Tracker.gradient((a)->critic(a), Params([crit_in]))[crit_in]
  
  act_grads = -crit_in.grad[end, :]
  zero_grad!(actor)
  Flux.back!(actions, act_grads)  # Chain rule
  opt_act()
  
  zero_grad!(critic)
  Flux.back!(loss_crit)
  opt_crit()

end

# --------------------------- Helper Functions --------------------------------

# stores the tuple of state, action, reward, next_state, and done
function remember(state, action, reward, next_state, done)
  if length(memory) >= MEM_SIZE
    deleteat!(memory, 1)
  end
  push!(memory, [state, action, reward, next_state, done])
end

# Choose action according to policy PendulumPolicy
function action(π::PendulumPolicy, reward, state, action)
  state = reshape(state, size(state)..., 1)
  act_pred = actor(state |> gpu).data +  ACTION_BOUND * sample_noise(ou)[1] * π.train
  clamp.(act_pred[:, 1], -ACTION_BOUND, ACTION_BOUND) |> cpu # returns action
end

function episode!(env, π = RandomPolicy())
  global frames
  ep = Episode(env, π) # Runs an episode with policy π
  frm = 0
  for (s, a, r, s′) in ep
    #OpenAIGym.render(env)
    r = env.done ? -1 : r
    if π.train remember(s, a, r, s′, env.done) end
    frames += 1
    frm += 1

    if length(memory) >= BATCH_SIZE && π.train
      train()
      update_target!(actor_target, actor; τ = τ)
      update_target!(critic_target, critic; τ = τ)
    end

    #frm > MAX_EP_LEN && break
  end
  
  ep.total_reward
end

# ------------------------------ Training --------------------------------------
scores = zeros(100)
e = 1
idx = 1
while e <= MAX_EP
  reset!(env)
  total_reward = episode!(env, PendulumPolicy())
  scores[idx] = total_reward
  idx = idx % 100 + 1
  avg = mean(scores)
  println("Episode: $e | Score: $total_reward | Avg score: $avg | Frames: $frames")
  e += 1
end

#=
# -------------------------------- Testing -------------------------------------
ee = 1

while true
  reset!(env)
  total_reward = episode!(env, PendulumPolicy(false))
  println("Episode: $ee | Score: $total_reward")
  ee += 1
end=#
