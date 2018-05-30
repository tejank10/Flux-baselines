using Flux
using OpenAIGym
import Reinforce.action
import Flux.params
using DiffEqNoiseProcess
using DiffEqNoiseProcess: NoiseProblem, solve

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
ACTION_BOUND = env.actions.hi[1]
BATCH_SIZE = 32
MEM_SIZE = 1000000
γ = 0.99    # discount rate
τ = 0.001 # for running average while updating target networks
η = 2.5e-4   # Learning rate

MAX_FRAMES = 1000000

memory = []

frames = 1

# -------------------------------- Action Noise --------------------------------

noise_proc = OrnsteinUhlenbeckProcess(0.15, zeros(ACTION_SIZE), 0.3, 0.0, zeros(ACTION_SIZE))
dt = 0.01
prob = NoiseProblem(noise_proc, (0, MAX_FRAMES * dt))
noise = solve(prob; dt = dt)

# ----------------------------- Model Architecture -----------------------------

actor = Chain(Dense(STATE_SIZE, 24, relu), Dense(24, 24, relu), Dense(24, ACTION_SIZE, tanh),
              x -> x * ACTION_BOUND)
actor_target = deepcopy(actor)

act_crit = Dense(ACTION_SIZE, 24)
state_crit = Chain(Dense(STATE_SIZE, 16, relu), Dense(16, 24))

critic = Chain(x -> state_crit(x[1:STATE_SIZE, :]) .+ act_crit(x[1 + STATE_SIZE:1 + STATE_SIZE, :]),
               x -> relu.(x), Dense(24, 1))
critic_target = deepcopy(critic)

# ------------------------------- Param Update Functions---------------------------------

function update_target!(target, model; τ = 0)
  for (p_t, p_m) in zip(params(target), params(model))
    p_t.data .= τ * p_t.data .+ (1 - τ) * p_m.data
  end
end

function nullify_grad!(p)
	if typeof(p) <: TrackedArray
		p.grad .= 0.
	end
	return p
end

function zero_grad!(model)
	model = mapleaves(nullify_grad!, model)
end

# ---------------------------------- Training ----------------------------------

opt_crit = ADAM(params(critic), η)
opt_act = ADAM(params(actor), η)

function train()
  # Getting data in shape
  minibatch = sample(memory, BATCH_SIZE)
	x = hcat(minibatch...)

  s = hcat(x[1, :]...)
  a = hcat(x[2, :]...)
  r = hcat(x[3, :]...)
  s′ = hcat(x[4, :]...)
  s_mask = .!hcat(x[5, :]...)

  # Update Critic
  a′ = actor_target(s′)
  crit_tgt_in = vcat(s′, a′.data)
  v′ = critic_target(crit_tgt_in).data
  y = r + γ * v′ .* s_mask	# set v′ to 0 where s_ is terminal state

  crit_in = vcat(s, a)
  v = critic(crit_in)
  loss_crit = Flux.mse(y, v)

  Flux.back!(loss_crit)
  opt_crit()

  # Update Actor
  actions = actor(s)
  crit_in = param(vcat(s, actions.data))
  Flux.back!(sum(critic(crit_in)))
  zero_grad!(critic)

  act_grads = -crit_in.grad[end, :]
  Flux.back!(sum(act_grads .* actions))  # Chain rule
  opt_act()
end

# --------------------------- Helper Functions --------------------------------

# stores the tuple of state, action, reward, next_state, and done
function remember(state, action, reward, next_state, done)
  if length(memory) >= MEM_SIZE
    deleteat!(memory, 1)
  end
  push!(memory, [state, action, reward, next_state, done])
end

# Choose action according to policy CartPolePolicy
function action(π::PendulumPolicy, reward, state, action)
  act_pred = actor(state) +  noise[frames] * π.train
  return clamp.(act_pred.data, -ACTION_BOUND, ACTION_BOUND) # returns action
end

function episode!(env, π = RandomPolicy())
  global frames
  ep = Episode(env, π) # Runs an episode with policy π

  for (s, a, r, s′) in ep
    OpenAIGym.render(env)
    r = env.done ? -1 : r
    if π.train remember(s, a, r, s′, env.done) end
    frames += 1

    if length(memory) >= BATCH_SIZE && π.train
      train()
      update_target!(actor_target, actor; τ = τ)
      update_target!(critic_target, critic; τ = τ)
    end
  end

  ep.total_reward
end

# ------------------------------ Training --------------------------------------

e = 1
while frames < MAX_FRAMES
  reset!(env)
  total_reward = episode!(env, PendulumPolicy())
  println("Episode: $e | Score: $total_reward")
  e += 1
end

# -------------------------------- Testing -------------------------------------
ee = 1

while true
  reset!(env)
  total_reward = episode!(env, PendulumPolicy(false))
  println("Episode: $ee | Score: $total_reward")
  ee += 1
end
