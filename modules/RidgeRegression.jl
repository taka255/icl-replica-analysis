module RidgeRegression

using LinearAlgebra, Statistics, Random

"""
    solve_rr(X, y; lambda=1.0, mean_loss=true, bias=true)

Returns the analytical solution for squared error + L2 regularization (Ridge regression).
"""
function solve_rr(
    X::AbstractMatrix, y::AbstractVector;
    lambda::Float64=1.0, mean_loss=true, bias=true
)
    N, D = size(X)
    reg = mean_loss ? N * lambda : lambda

    if bias
        x_mean = mean(X, dims=1)
        y_mean = mean(y)
        Xc = X .- x_mean      
        yc = y .- y_mean      

        # (Xc' * Xc + reg * I(D))の固有値をorint
        println("Eigenvalues of (Xc' * Xc + reg * I(D)): ", eigvals(Xc' * Xc + reg * I(D)))
        w = (Xc' * Xc + reg * I(D)) \ (Xc' * yc)
        b = y_mean[] - (x_mean * w)[1]
    else
        w = (X' * X + reg * I(D)) \ (X' * y)
        b = zero(eltype(y))
    end

    return bias ? (w, b) : w
end




"""
    solve_rr_gd(X, y; lambda=1.0, mean_loss=true, bias=true, eta=1e-4, n_iter=50000, rng=Random.GLOBAL_RNG)

Optimizes the same Ridge regression objective function using gradient descent.
- eta: learning rate
- n_iter: maximum number of iterations
- tol: tolerance for convergence (stop when loss improvement < tol)
- print_loss: whether to print loss during training
- rng: random number generator for initializing parameters
"""
function solve_rr_gd(
    X::AbstractMatrix, y::AbstractVector;
    lambda::Float64=1.0, mean_loss=true, bias=true,
    eta::Float64=1e-5, n_iter::Int=10000, tol::Float64=1e-8, print_loss=false, rng=Random.GLOBAL_RNG
)
    N, D = size(X)
    reg = mean_loss ? N * lambda : lambda

    w = zeros(D)
    if rng !== :zero_init
        w .= randn(rng, D) 
    end

    b = bias ? randn(rng) : 0.0
    
    # pre-allocate vectors for reuse
    grad_w = zeros(D)
    y_pred = similar(y)
    error = similar(y)
    
    # pre-compute matrix transpose (once)
    Xt = X'
    
    prev_loss = Inf
    converged = false

    for i in 1:n_iter
        # y_pred = X * w .+ b
        mul!(y_pred, X, w)
        y_pred .+= b
        
        # error = y_pred - y
        error .= y_pred .- y

        # grad_w = X' * error + reg * w
        mul!(grad_w, Xt, error)
        grad_w .+= reg .* w
        
        w .-= eta .* grad_w
        
        if bias
            grad_b = sum(error)
            b -= eta * grad_b
        end

        # convergence check
        if i % 100 == 0 || i == n_iter
            current_loss = 0.5 * dot(error, error) + 0.5 * reg * dot(w, w)
            
            if abs(prev_loss - current_loss) < tol
                converged = true
                if print_loss
                    println("Converged at iteration $i with loss: $current_loss")
                end
                break
            end
            
            prev_loss = current_loss
            
            if print_loss && i % 20000 == 0
                println("Iter $i / $n_iter, Loss: $current_loss")
            end
        end
    end
    
    if !converged && print_loss
        @warn "Did not converge within $n_iter iterations"
    end

    return bias ? (w, b) : w
end


export solve_rr, solve_rr_gd

end # module RidgeRegression
