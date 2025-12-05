include("modules/Experiments.jl")
using .Experiments: generate_data, calc_generalization_error, calc_orderparams
using Flux, Optimisers
using HDF5


D = 20
α = 6.0
τ = 1.5   # τ = γ * κ
κ = 0.5
σ = 0.01
ρs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
num_trials = 5

L  = round(Int, D * α)
M  = round(Int, D^2 * τ)
M0 = round(Int, D * κ)

in_dim = D + 1
#d_attn = D + 1 # This might be too large; how about setting it to 4?
d_attn = 4
T      = L + 1


# ==========================
# one-layer softmax attention
# ==========================

struct OneLayerSelfAttention
    W_Q::Dense
    W_K::Dense
    W_V::Dense   # Output dim D+1 to avoid W_O
end

Flux.@functor OneLayerSelfAttention

function OneLayerSelfAttention(in_dim::Int, d_attn::Int)
    W_Q = Dense(in_dim, d_attn, bias=false)
    W_K = Dense(in_dim, d_attn, bias=false)
    W_V = Dense(in_dim, in_dim, bias=false)  # Set to in_dim to omit W_O
    return OneLayerSelfAttention(W_Q, W_K, W_V)
end

# Forward for single sample: x :: (in_dim, T) → predicted scalar
function (m::OneLayerSelfAttention)(x::AbstractMatrix)
    # x: (in_dim, T)
    x_last = x[:, end:end]             # (in_dim, 1) - last token
    q_last = m.W_Q(x_last)[:, 1]       # (d_attn,) - last query only

    K = m.W_K(x)                       # (d_attn, T)
    V = m.W_V(x)                       # (in_dim, T)
    d_attn, T = size(K)

    scores = (q_last' * K) ./ sqrt(Float32(d_attn))  # (1, T)
    attn   = Flux.softmax(scores; dims=2)            # (1, T)

    # Readout only from label dimension (last)
    return (V[end, :]' * attn')[1]     # scalar
end

# ==========================
# Preprocessing: rearrange C once and batch
# ==========================


nepochs   = 3000
tolerance = 1e-5


# ==========================
# Batch loss: compute Dense in one go
# ==========================
function loss(m::OneLayerSelfAttention, X3_local::AbstractArray, y_true_local::AbstractVector, in_dim_local::Int, d_attn_local::Int, T_local::Int, M_local::Int)
    @assert size(X3_local) == (in_dim_local, T_local, M_local)

    # Reshape to (in_dim, T*M) (view, no copy)
    Xflat = reshape(X3_local, in_dim_local, T_local*M_local)      # (in_dim, T*M)

    # Compute Q,K,V for all μ at once
    Qflat = m.W_Q(Xflat)                  # (d_attn, T*M)
    Kflat = m.W_K(Xflat)                  # (d_attn, T*M)
    Vflat = m.W_V(Xflat)                  # (in_dim, T*M)

    # Reshape back to 3D (view)
    Q = reshape(Qflat, d_attn_local, T_local, M_local)
    K = reshape(Kflat, d_attn_local, T_local, M_local)
    V = reshape(Vflat, in_dim_local, T_local, M_local)

    inv_sqrt_d_attn = 1.0f0 / sqrt(Float32(d_attn_local))

    # Build prediction y_hat[μ] for each μ functionally
    y_hat = map(1:M_local) do μ
        q_last = @view Q[:, T_local, μ]        # (d_attn,)
        Kμ     = @view K[:, :, μ]        # (d_attn, T)
        Vlast  = @view V[end, :, μ]      # (T,)

        scores = (q_last' * Kμ) .* inv_sqrt_d_attn   # (1, T)
        attn   = Flux.softmax(scores; dims=2)        # (1, T)

        Float32((Vlast' * attn')[1])                 # scalar
    end

    diff = y_hat .- y_true_local
    return sum(diff .* diff) / M_local
end



function single_experiment(D, L, M, M0, σ, r)
    X, W0, W, Y, _, A = generate_data(D, L, M, M0, σ, r)
    C = zeros(Float32, M, L+1, D+1)
    for μ in 1:M
        C[μ, :, 1:D] .= Float32.(X[μ, :, :])
        C[μ, :, D+1] .= Float32.(Y[μ, :])
        C[μ, L+1, D+1] = 0.0f0
    end
    X3_local = permutedims(C, (3, 2, 1))  # (in_dim, T, M)
    y_true_local = Float32.(Y[:, L+1])    # (M,)
    
    model = OneLayerSelfAttention(in_dim, d_attn)
    opt = Optimisers.setup(Optimisers.Adam(1e-3), model)
    
    previous_loss = Inf
    for epoch in 1:nepochs
        l, back = Flux.withgradient(model) do m
            loss(m, X3_local, y_true_local, in_dim, d_attn, T, M)
        end
        Optimisers.update!(opt, model, back[1])
        
        loss_change = abs(previous_loss - l)
        
        if epoch % 10 == 0 || loss_change < tolerance
            @info "epoch = $epoch, loss = $(l), loss_change = $(loss_change)"
        end
        
        if loss_change < tolerance
            @info "Training converged (tolerance = $tolerance)"
            break
        end
        
        previous_loss = l
    end
    
    return model, opt, W0, A
end

# ==========================
# Verification (single sample)
# ==========================
# ==========================
# system & experimental params
# ==========================

indexes = length(ρs) * num_trials


println("num_trials = $num_trials")

Threads.@threads for ind in 1:indexes
    println("ind = $ind")
    i = mod(ind - 1, length(ρs)) + 1  # 1-based indexing
    j = div(ind - 1, length(ρs)) + 1
    ρ = ρs[i]
    r = round(Int, D * ρ)
    model, opt, W0, A = single_experiment(D, L, M, M0, σ, r)
    
    # Extract model parameters as arrays
    W_Q_weight = model.W_Q.weight
    W_K_weight = model.W_K.weight
    W_V_weight = model.W_V.weight
    
    
    h5open("data_smalldin/nonlinear_attention_d_attn$(d_attn)_D$(D)_L$(L)_M$(M)_M0$(M0)_ρ$(ρ)_trial$(j).h5", "w") do file
        write(file, "W_Q_weight", W_Q_weight)
        write(file, "W_K_weight", W_K_weight)
        write(file, "W_V_weight", W_V_weight)
        # Save metadata as well
        write(file, "d_attn", d_attn)
        write(file, "D", D)
        write(file, "L", L)
        write(file, "M", M)
        write(file, "M0", M0)
        write(file, "ρ", ρ)
        write(file, "r", r)
        write(file, "trial", j)
        write(file, "W0", W0)
        write(file, "A", A)
    end
    
    println("ind = $ind, done")
end



println("all done")