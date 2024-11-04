using OptimalControl
using NLPModelsIpopt
using Plots

const t0 = 0      # initial time
const h0 = 1      # initial altitude
const v0 = 0      # initial speed
const m0 = 1      # initial mass
const mf = 0.6    # final mass to target
const Tmax = 3.5

@def ocp begin # definition of the optimal control problem

    tf ∈ R, variable
    t ∈ [ t0, tf ], time
    x = (h, v, m) ∈ R³, state
    u ∈ R, control

    x(t0) == [ h0, v0, m0 ]
    m(tf) == mf
    0 ≤ u(t) ≤ Tmax
    h(t) ≥ h0
    q(h(t), v(t)) ≤ 0

    ẋ(t) == F0(x(t)) + u(t) * F1(x(t))

    h(tf) → max

end;

# Dynamics
const Cd = 310
const β = 500
const b = 2
const qmax = 10

d(h, v) = Cd * v^2 * exp(-β*(h - 1)) # drag force
q(h, v) = 20 * d(h, v) - qmax        # dynamic pressure

F0(h, v, m) = [ v, -d(h, v)/m - 1/h^2,  0 ]
F1(h, v, m) = [ 0,                1/m, -b ]

F0(x) = F0(x...)
F1(x) = F1(x...)

args(grid_size, print_level, tol) = isnothing(tol) ? (grid_size=grid_size, print_level=print_level) : (grid_size=grid_size, print_level=print_level, tol=tol)
tol = 1e-7

sol = solve(ocp; args(50, 5, tol)...)
plt = plot(sol, solution_label="(direct)", size=(800, 800))

sol = solve(ocp; init=sol, args(500, 5, tol)...)
plt = plot(sol, solution_label="(direct)", size=(800, 800))

sol = solve(ocp; init=sol, args(1000, 5, tol)...)
plt = plot(sol, solution_label="(direct)", size=(800, 800))
