using HDF5, CairoMakie, Flux, Optimisers, Statistics
include("modules/Experiments.jl")
using .Experiments: generate_data


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


# Load data
trial_nums = [1, 2, 3, 4, 5]
rhos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
out_of_distribution_gen_errors = zeros(length(rhos), length(trial_nums))
memorization_gen_errors = zeros(length(rhos), length(trial_nums))
in_distribution_gen_errors = zeros(length(rhos), length(trial_nums))

σ = 0.01

# plot 1
for (i, rho) in enumerate(rhos)
    for (j, trial_num) in enumerate(trial_nums)
        d_attn = 4#20+1
        path = "data_smalldin/nonlinear_attention_d_attn4_D20_L40_M600_M010_ρ$(rho)_trial$(trial_num).h5"
        h5open(path, "r") do file
            W_Q_weight = read(file, "W_Q_weight")
            W_K_weight = read(file, "W_K_weight")
            W_V_weight = read(file, "W_V_weight")
            W0_read = read(file, "W0")
            A_read = read(file, "A")

            D = read(file, "D")
            L = read(file, "L")
            M = read(file, "M")
            M0 = read(file, "M0")
            ρ = read(file, "ρ")
            r = read(file, "r")
            trial = read(file, "trial")

            # Reconstruct trained model
            in_dim = D + 1
            
            model = OneLayerSelfAttention(in_dim, d_attn)
            model.W_Q.weight .= W_Q_weight
            model.W_K.weight .= W_K_weight
            model.W_V.weight .= W_V_weight

            n_test = 3000



            # out-of-distribution task
            X, W0, W, Y, _, _ = generate_data(D, L, n_test, 10000, σ, D)
            C = zeros(Float32, n_test, L+1, D+1)
            for μ in 1:n_test
                C[μ, :, 1:D] .= Float32.(X[μ, :, :])
                C[μ, :, D+1] .= Float32.(Y[μ, :])
                C[μ, L+1, D+1] = 0.0f0
            end
            X3_local = permutedims(C, (3, 2, 1))  # (in_dim, T, n_test)
            y_true_local = Y[:, L+1]    # size: (n_test,)
            # Process each sample individually
            y_hat = [model(@view X3_local[:, :, μ]) for μ in 1:n_test]
            gen_error = sum((y_hat .- y_true_local) .^ 2) / n_test
            @info "gen_error = $(gen_error)"
            out_of_distribution_gen_errors[i, j] = gen_error

            
            # in-distribution task
            X, W0, W, Y, _, _ = generate_data(D, L, n_test, 10000, σ, r, A=A_read)
            C = zeros(Float32, n_test, L+1, D+1)
            for μ in 1:n_test
                C[μ, :, 1:D] .= Float32.(X[μ, :, :])
                C[μ, :, D+1] .= Float32.(Y[μ, :])
                C[μ, L+1, D+1] = 0.0f0
            end
            X3_local = permutedims(C, (3, 2, 1))  # (in_dim, T, n_test)
            y_true_local = Y[:, L+1]    # size: (n_test,)
            # Process each sample individually
            y_hat = [model(@view X3_local[:, :, μ]) for μ in 1:n_test]
            gen_error = sum((y_hat .- y_true_local) .^ 2) / n_test
            @info "gen_error = $(gen_error)"
            in_distribution_gen_errors[i, j] = gen_error



            # memorization task
            X, W0, W, Y, _, _ = generate_data(D, L, n_test, M0, σ, D, W0=W0_read)
            C = zeros(Float32, n_test, L+1, D+1)
            for μ in 1:n_test
                C[μ, :, 1:D] .= Float32.(X[μ, :, :])
                C[μ, :, D+1] .= Float32.(Y[μ, :])
                C[μ, L+1, D+1] = 0.0f0
            end
            X3_local = permutedims(C, (3, 2, 1))  # (in_dim, T, n_test)
            y_true_local = Y[:, L+1]    # size: (n_test,)
            # Process each sample individually
            y_hat = [model(@view X3_local[:, :, μ]) for μ in 1:n_test]
            gen_error = sum((y_hat .- y_true_local) .^ 2) / n_test
            @info "gen_error = $(gen_error)"
            memorization_gen_errors[i, j] = gen_error
            
        end
    end
end

out_of_distribution_gen_errors_mean = vec(mean(out_of_distribution_gen_errors, dims=2))
out_of_distribution_gen_errors_std = vec(std(out_of_distribution_gen_errors, dims=2)) ./ sqrt(length(trial_nums))
memorization_gen_errors_mean = vec(mean(memorization_gen_errors, dims=2))
memorization_gen_errors_std = vec(std(memorization_gen_errors, dims=2)) ./ sqrt(length(trial_nums))
in_distribution_gen_errors_mean = vec(mean(in_distribution_gen_errors, dims=2))
in_distribution_gen_errors_std = vec(std(in_distribution_gen_errors, dims=2)) ./ sqrt(length(trial_nums))



limits = (nothing, nothing, 0.0, 1.6)
fig = Figure(size = (1800, 500))
ax = Axis(fig[1, 1], xlabel=L"\rho", ylabel="Generalization Error", title=L"\alpha = 2.0", 
  titlesize=20, xlabelsize=20, limits=limits)
scatter!(ax, rhos, out_of_distribution_gen_errors_mean, label="ODG")
scatter!(ax, rhos, memorization_gen_errors_mean, label="TM")
scatter!(ax, rhos, in_distribution_gen_errors_mean, label="IDG")
lines!(ax, rhos, out_of_distribution_gen_errors_mean, label="ODG")
lines!(ax, rhos, memorization_gen_errors_mean, label="TM")
lines!(ax, rhos, in_distribution_gen_errors_mean, label="IDG")

errorbars!(ax, rhos, out_of_distribution_gen_errors_mean, out_of_distribution_gen_errors_std, whiskerwidth=10, label="ODG")
errorbars!(ax, rhos, memorization_gen_errors_mean, memorization_gen_errors_std, whiskerwidth=10, label="TM")
errorbars!(ax, rhos, in_distribution_gen_errors_mean, in_distribution_gen_errors_std, whiskerwidth=10, label="IDG")
axislegend(ax, position=:rb, merge=true)



# plot 2
for (i, rho) in enumerate(rhos)
    for (j, trial_num) in enumerate(trial_nums)
        d_attn = 4#20+1
        path = "data_smalldin/nonlinear_attention_d_attn4_D20_L80_M600_M010_ρ$(rho)_trial$(trial_num).h5"
        h5open(path, "r") do file
            W_Q_weight = read(file, "W_Q_weight")
            W_K_weight = read(file, "W_K_weight")
            W_V_weight = read(file, "W_V_weight")
            W0_read = read(file, "W0")
            A_read = read(file, "A")

            D = read(file, "D")
            L = read(file, "L")
            M = read(file, "M")
            M0 = read(file, "M0")
            ρ = read(file, "ρ")
            r = read(file, "r")
            trial = read(file, "trial")

            # 学習したmodelの再現
            in_dim = D + 1
            
            model = OneLayerSelfAttention(in_dim, d_attn)
            model.W_Q.weight .= W_Q_weight
            model.W_K.weight .= W_K_weight
            model.W_V.weight .= W_V_weight

            n_test = 3000



            # out-of-distribution task
            X, W0, W, Y, _, _ = generate_data(D, L, n_test, 10000, σ, D)
            C = zeros(Float32, n_test, L+1, D+1)
            for μ in 1:n_test
                C[μ, :, 1:D] .= Float32.(X[μ, :, :])
                C[μ, :, D+1] .= Float32.(Y[μ, :])
                C[μ, L+1, D+1] = 0.0f0
            end
            X3_local = permutedims(C, (3, 2, 1))  # (in_dim, T, n_test)
            y_true_local = Y[:, L+1]    # size: (n_test,)
            # Process each sample individually
            y_hat = [model(@view X3_local[:, :, μ]) for μ in 1:n_test]
            gen_error = sum((y_hat .- y_true_local) .^ 2) / n_test
            @info "gen_error = $(gen_error)"
            out_of_distribution_gen_errors[i, j] = gen_error

            
            # in-distribution task
            X, W0, W, Y, _, _ = generate_data(D, L, n_test, 10000, σ, r, A=A_read)
            C = zeros(Float32, n_test, L+1, D+1)
            for μ in 1:n_test
                C[μ, :, 1:D] .= Float32.(X[μ, :, :])
                C[μ, :, D+1] .= Float32.(Y[μ, :])
                C[μ, L+1, D+1] = 0.0f0
            end
            X3_local = permutedims(C, (3, 2, 1))  # (in_dim, T, n_test)
            y_true_local = Y[:, L+1]    # size: (n_test,)
            # Process each sample individually
            y_hat = [model(@view X3_local[:, :, μ]) for μ in 1:n_test]
            gen_error = sum((y_hat .- y_true_local) .^ 2) / n_test
            @info "gen_error = $(gen_error)"
            in_distribution_gen_errors[i, j] = gen_error



            # memorization task
            X, W0, W, Y, _, _ = generate_data(D, L, n_test, M0, σ, D, W0=W0_read)
            C = zeros(Float32, n_test, L+1, D+1)
            for μ in 1:n_test
                C[μ, :, 1:D] .= Float32.(X[μ, :, :])
                C[μ, :, D+1] .= Float32.(Y[μ, :])
                C[μ, L+1, D+1] = 0.0f0
            end
            X3_local = permutedims(C, (3, 2, 1))  # (in_dim, T, n_test)
            y_true_local = Y[:, L+1]    # size: (n_test,)
            # Process each sample individually
            y_hat = [model(@view X3_local[:, :, μ]) for μ in 1:n_test]
            gen_error = sum((y_hat .- y_true_local) .^ 2) / n_test
            @info "gen_error = $(gen_error)"
            memorization_gen_errors[i, j] = gen_error
            
        end
    end
end

out_of_distribution_gen_errors_mean = vec(mean(out_of_distribution_gen_errors, dims=2))
out_of_distribution_gen_errors_std = vec(std(out_of_distribution_gen_errors, dims=2)) ./ sqrt(length(trial_nums))
memorization_gen_errors_mean = vec(mean(memorization_gen_errors, dims=2))
memorization_gen_errors_std = vec(std(memorization_gen_errors, dims=2)) ./ sqrt(length(trial_nums))
in_distribution_gen_errors_mean = vec(mean(in_distribution_gen_errors, dims=2))
in_distribution_gen_errors_std = vec(std(in_distribution_gen_errors, dims=2)) ./ sqrt(length(trial_nums))





ax = Axis(fig[1, 2], xlabel=L"\rho", ylabel="Generalization Error", title=L"\alpha = 4.0", 
   titlesize=20, xlabelsize=20, limits=limits)
scatter!(ax, rhos, out_of_distribution_gen_errors_mean, label="ODG")
scatter!(ax, rhos, memorization_gen_errors_mean, label="TM")
scatter!(ax, rhos, in_distribution_gen_errors_mean, label="IDG")
lines!(ax, rhos, out_of_distribution_gen_errors_mean, label="ODG")
lines!(ax, rhos, memorization_gen_errors_mean, label="TM")
lines!(ax, rhos, in_distribution_gen_errors_mean, label="IDG")
errorbars!(ax, rhos, out_of_distribution_gen_errors_mean, out_of_distribution_gen_errors_std, whiskerwidth=10, label="ODG")
errorbars!(ax, rhos, memorization_gen_errors_mean, memorization_gen_errors_std, whiskerwidth=10, label="TM")
errorbars!(ax, rhos, in_distribution_gen_errors_mean, in_distribution_gen_errors_std, whiskerwidth=10, label="IDG")
axislegend(ax, position=:rb, merge=true)







# plot 3
for (i, rho) in enumerate(rhos)
    for (j, trial_num) in enumerate(trial_nums)
        d_attn = 4#20+1
        path = "data_smalldin/nonlinear_attention_d_attn4_D20_L120_M600_M010_ρ$(rho)_trial$(trial_num).h5"
        h5open(path, "r") do file
            W_Q_weight = read(file, "W_Q_weight")
            W_K_weight = read(file, "W_K_weight")
            W_V_weight = read(file, "W_V_weight")
            W0_read = read(file, "W0")
            A_read = read(file, "A")

            D = read(file, "D")
            L = read(file, "L")
            M = read(file, "M")
            M0 = read(file, "M0")
            ρ = read(file, "ρ")
            r = read(file, "r")
            trial = read(file, "trial")

            # 学習したmodelの再現
            in_dim = D + 1
            
            model = OneLayerSelfAttention(in_dim, d_attn)
            model.W_Q.weight .= W_Q_weight
            model.W_K.weight .= W_K_weight
            model.W_V.weight .= W_V_weight

            n_test = 3000



            # out-of-distribution task
            X, W0, W, Y, _, _ = generate_data(D, L, n_test, 10000, σ, D)
            C = zeros(Float32, n_test, L+1, D+1)
            for μ in 1:n_test
                C[μ, :, 1:D] .= Float32.(X[μ, :, :])
                C[μ, :, D+1] .= Float32.(Y[μ, :])
                C[μ, L+1, D+1] = 0.0f0
            end
            X3_local = permutedims(C, (3, 2, 1))  # (in_dim, T, n_test)
            y_true_local = Y[:, L+1]    # size: (n_test,)
            # Process each sample individually
            y_hat = [model(@view X3_local[:, :, μ]) for μ in 1:n_test]
            gen_error = sum((y_hat .- y_true_local) .^ 2) / n_test
            @info "gen_error = $(gen_error)"
            out_of_distribution_gen_errors[i, j] = gen_error

            
            # in-distribution task
            X, W0, W, Y, _, _ = generate_data(D, L, n_test, 10000, σ, r, A=A_read)
            C = zeros(Float32, n_test, L+1, D+1)
            for μ in 1:n_test
                C[μ, :, 1:D] .= Float32.(X[μ, :, :])
                C[μ, :, D+1] .= Float32.(Y[μ, :])
                C[μ, L+1, D+1] = 0.0f0
            end
            X3_local = permutedims(C, (3, 2, 1))  # (in_dim, T, n_test)
            y_true_local = Y[:, L+1]    # size: (n_test,)
            # Process each sample individually
            y_hat = [model(@view X3_local[:, :, μ]) for μ in 1:n_test]
            gen_error = sum((y_hat .- y_true_local) .^ 2) / n_test
            @info "gen_error = $(gen_error)"
            in_distribution_gen_errors[i, j] = gen_error



            # memorization task
            X, W0, W, Y, _, _ = generate_data(D, L, n_test, M0, σ, D, W0=W0_read)
            C = zeros(Float32, n_test, L+1, D+1)
            for μ in 1:n_test
                C[μ, :, 1:D] .= Float32.(X[μ, :, :])
                C[μ, :, D+1] .= Float32.(Y[μ, :])
                C[μ, L+1, D+1] = 0.0f0
            end
            X3_local = permutedims(C, (3, 2, 1))  # (in_dim, T, n_test)
            y_true_local = Y[:, L+1]    # size: (n_test,)
            # Process each sample individually
            y_hat = [model(@view X3_local[:, :, μ]) for μ in 1:n_test]
            gen_error = sum((y_hat .- y_true_local) .^ 2) / n_test
            @info "gen_error = $(gen_error)"
            memorization_gen_errors[i, j] = gen_error
            
        end
    end
end

out_of_distribution_gen_errors_mean = vec(mean(out_of_distribution_gen_errors, dims=2))
out_of_distribution_gen_errors_std = vec(std(out_of_distribution_gen_errors, dims=2)) ./ sqrt(length(trial_nums))
memorization_gen_errors_mean = vec(mean(memorization_gen_errors, dims=2))
memorization_gen_errors_std = vec(std(memorization_gen_errors, dims=2)) ./ sqrt(length(trial_nums))
in_distribution_gen_errors_mean = vec(mean(in_distribution_gen_errors, dims=2))
in_distribution_gen_errors_std = vec(std(in_distribution_gen_errors, dims=2)) ./ sqrt(length(trial_nums))





ax = Axis(fig[1, 3], xlabel=L"\rho", ylabel="Generalization Error", title=L"\alpha = 6.0", 
   titlesize=20, xlabelsize=20, limits=limits)
scatter!(ax, rhos, out_of_distribution_gen_errors_mean, label="ODG")
scatter!(ax, rhos, memorization_gen_errors_mean, label="TM")
scatter!(ax, rhos, in_distribution_gen_errors_mean, label="IDG")
lines!(ax, rhos, out_of_distribution_gen_errors_mean, label="ODG")
lines!(ax, rhos, memorization_gen_errors_mean, label="TM")
lines!(ax, rhos, in_distribution_gen_errors_mean, label="IDG")
errorbars!(ax, rhos, out_of_distribution_gen_errors_mean, out_of_distribution_gen_errors_std, whiskerwidth=10, label="ODG")
errorbars!(ax, rhos, memorization_gen_errors_mean, memorization_gen_errors_std, whiskerwidth=10, label="TM")
errorbars!(ax, rhos, in_distribution_gen_errors_mean, in_distribution_gen_errors_std, whiskerwidth=10, label="IDG")
axislegend(ax, position=:rb, merge=true)
display(fig)

save("assets/nonlinear_attention_gen_error.png", fig)












