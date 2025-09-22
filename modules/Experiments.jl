module Experiments

using Random, LinearAlgebra, Statistics, Distributions


function generate_data(D::Int, L::Int, M::Int, M0::Int, σ::Float64, r::Int; rng=Random.GLOBAL_RNG)

    # allocate output arrays
    X = Array{Float64,3}(undef, M, L+1, D)
    W0 = Array{Float64,2}(undef, M0, D)
    W = Array{Float64,2}(undef, M, D)
    Y = Array{Float64,2}(undef, M, L+1)
    H = Array{Float64,3}(undef, M, D, D)


    # generate W_0
    A = randn(rng, D, r) 
    A = Matrix(qr(A).Q)
    for μ0 in 1:M0
        v = randn(rng, r)
        w = A * v * sqrt(D)/sqrt(r)
        W0[μ0, :] .= w
    end


    # generate W (select M tasks uniformly at random)
    task_indices = rand(rng, 1:M0, M)
    for μ in 1:M
        W[μ, :] = W0[task_indices[μ], :]
    end


    # generate X, Y, H
    for μ in 1:M
        sum_yxT_row = zeros(D)

        for l in 1:L
            X[μ, l, :] = randn(rng, D) ./ sqrt(D)
            Y[μ, l] = @views dot(W[μ, :], X[μ, l, :]) + σ * randn(rng) 
            @views sum_yxT_row .+= @views Y[μ, l] .* X[μ, l, :]
        end

        # for query and label
        X[μ, L+1, :] = randn(rng, D) ./ sqrt(D)
        Y[μ, L+1] = @views dot(W[μ, :], X[μ, L+1, :]) + σ * randn(rng)

        # generate H
        H[μ, :, :] = @views (D / L) * (X[μ, L+1, :] * sum_yxT_row')  # (D×1)×(1×D) → D×D
    end

    return X, W0, W, Y, H, A
end



function convert_to_vec(H::Array{Float64,3})
    M, D, D = size(H)
    vecH = zeros(M, D*D)
    for μ in 1:M
        vecH[μ, :] = vec(H[μ, :, :])
    end
    return vecH
end



function calc_generalization_error(W0_in, vec_Γ, A_in, L_tilde, r, σ_test;
                                  n_test=10, n_samples_per_task=1, rng=Random.GLOBAL_RNG, error_type=:memory)

    M0, D = size(W0_in)
    total_gen_errors = zeros(n_test)

    # allocate arrays for task generation
    W0 = similar(W0_in)
    A = copy(A_in) # for :IDG
    local v, A_new # for :ODG
    
    if error_type == :id
        v = zeros(M0, r)
    elseif error_type == :ood
        v = zeros(M0, r)
        A_new = similar(A_in, D, r)
    end

    # allocate arrays for data generation and calculation
    sum_yxT_row = zeros(D)
    X_sample = zeros(D)
    X_test = zeros(D)
    H = zeros(D, D)

    # for :memory
    if error_type == :memory
        W0 .= W0_in
    end

    # main loop
    for test_run in 1:n_test
        
        # generate task (W0)
        if error_type == :id
            randn!(rng, v)
            mul!(W0, v, A') # W0 = v * A'
            W0 .*= (sqrt(D) / sqrt(r))
        elseif error_type == :ood
            # generate new random orthogonal matrix A_new
            copyto!(A_new, Matrix(qr(randn(rng, D, r)).Q))
            
            randn!(rng, v)
            mul!(W0, v, A_new') # W0 = v * A_new'
            W0 .*= (sqrt(D) / sqrt(r))
        end

        # calculate generalization error
        sum_of_errors_in_run = 0.0
        for i in 1:M0
            w = @views W0[i, :]
            task_error_sum = 0.0

            for _ in 1:n_samples_per_task
                # reset arrays
                fill!(sum_yxT_row, 0.0)

                # generate data (L samples)
                for l in 1:L_tilde
                    randn!(rng, X_sample)
                    X_sample ./= sqrt(D)
                    y_sample = dot(w, X_sample) + σ_test * randn(rng)
                    sum_yxT_row .+= y_sample .* X_sample
                end

                # generate test data (L+1-th sample)
                randn!(rng, X_test)
                X_test ./= sqrt(D)
                Y_test = dot(w, X_test) + σ_test * randn(rng)

                # calculate H matrix (avoid allocation by outer product and broadcast)
                H .= (D / L_tilde) .* X_test .* sum_yxT_row'
                
                y_est = dot(vec_Γ, vec(H))
                task_error_sum += (Y_test - y_est)^2
            end
            
            sum_of_errors_in_run += task_error_sum / n_samples_per_task
        end
        
        total_gen_errors[test_run] = sum_of_errors_in_run / M0
    end

    return mean(total_gen_errors)
end



function calc_orderparams(W0, Γ; sigmas=nothing, A=nothing, r=nothing)
    if sigmas === nothing
        sigmas = ones(size(W0, 1))
    end

    M0, D = size(W0)

    # Q_bar, m_bar
    Q_bar = tr(Γ' * Γ) / D
    m_bar = tr(Γ) / D

    # Q, m
    Q = 0.0
    m = 0.0
    for i in 1:M0 
        w = W0[i, :] / sigmas[i]
        Q += dot(Γ * w, Γ * w) / D
        m += dot(w, Γ * w) / D
    end
    Q /= M0
    m /= M0

    # Q0, m0
    if A !== nothing && r !== nothing
        Q0 = tr((Γ * A)' * (Γ * A)) / r
        m0 = tr(A' * Γ * A) / r
    else
        Q0 = Q
        m0 = m
    end

    return Q, m, Q_bar, m_bar, Q0, m0
end    





export generate_data, convert_to_vec, calc_generalization_error, calc_orderparams

end
