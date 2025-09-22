module Replica

using LinearAlgebra, SpecialFunctions, Distributions


###############################################################################
# Stieltjes transform and its derivative
###############################################################################

function gS_lowrank(z::Float64, κ::Float64, ρ::Float64)
    sqrt_arg = (ρ*κ*z + ρ - κ)^2 - 4*ρ^2*z*κ
    numerator = κ + ρ - κ*ρ*z - 2 - sqrt(sqrt_arg) 
    denominator = 2*z
    if sqrt_arg < 0
        return NaN
    end
    return numerator / denominator
end

function gS_prime_lowrank(z::Float64, κ::Float64, ρ::Float64, gS_val_in::Float64)
    if z == 0.0
        return NaN
    end
    
    numerator_of_fraction = ρ * κ * (ρ*κ*z - ρ - κ) 
    denominator_of_fraction = κ + ρ - 2.0 - κ*ρ*z - 2.0*z*gS_val_in
    
    if denominator_of_fraction == 0.0
        return Inf
    end
    
    fraction = numerator_of_fraction / denominator_of_fraction
    numerator_of_g_prime = -κ*ρ - 2.0*gS_val_in - fraction
    g_prime = numerator_of_g_prime / (2.0*z)
    
    return g_prime
end

function gS_tilde(z::Float64, κ::Float64, ρ::Float64)

    A = κ * ρ * z + ρ - κ
    B_sqrt = sqrt(A^2 - 4 * ρ^2 * κ * z)
    
    numerator = -A - B_sqrt
    denominator = 2 * ρ * z
    
    return numerator / denominator
end

function gS_prime_tilde(z::Float64, κ::Float64, ρ::Float64)
    # first, calculate g(z)
    g_val = gS_tilde(z, κ, ρ)
    
    numerator = -ρ * g_val * (g_val + κ)
    denominator = 2 * ρ * z * g_val + κ * ρ * z + ρ - κ
    
    return numerator / denominator
end





###############################################################################
#  replica update
###############################################################################

function solve_replica(α::Float64, τ::Float64, κ::Float64, λ::Float64, σ::Float64, ρ::Float64;
    max_iter::Int=1000, tol::Float64=1e-9, damp::Float64 = 0.5, decomposed::Bool = false)

    # initial values of order parameters
    q::Float64 = 1.0
    m::Float64 = 1.0
    χ::Float64 = 1.0
    q_bar::Float64 = 1.0
    chi_bar::Float64 = 1.0

    # variables used in the loop
    local hat_q::Float64, hat_bar_q::Float64, hat_m::Float64, hat_chi::Float64, hat_bar_chi::Float64
    local z_val::Float64, gs_val::Float64, gs_prime_val::Float64
    local m_bar::Float64, m0::Float64, q0::Float64
    local q_temp::Float64, m_temp::Float64, chi_temp::Float64, q_bar_temp::Float64, chi_bar_temp::Float64


    for iter in 1:max_iter
        q_old = q
        m_old = m
        χ_old = χ
        q_bar_old = q_bar
        chi_bar_old = chi_bar


        # update hat parameters
        common_den_val = 1 + χ + (1/α) * (1 + σ^2) * chi_bar
        hat_q = (τ/κ) / common_den_val 
        hat_bar_q = (τ/(α*κ)) * (1 + σ^2) / common_den_val
        hat_m = (τ/κ) / common_den_val 

        common_num_chi_val = (1/α)*(1 + σ^2)*q_bar + q - 2*m + 1 + σ^2  
        hat_chi = (τ/κ) * common_num_chi_val / common_den_val^2 
        hat_bar_chi = (τ/(α*κ)) * (1 + σ^2) * common_num_chi_val / common_den_val^2

        # update order parameters
        z_val = -(λ + hat_bar_q) / hat_q
        gs_val = gS_lowrank(z_val, κ, ρ)
        gs_prime_val = gS_prime_lowrank(z_val, κ, ρ, gs_val)

        # q
        term_q_T = (hat_chi * (1.0 + 2*z_val*gs_val + z_val^2*gs_prime_val) / (κ) ) / (hat_q^2)
        term_q_S = (hat_m^2 * (1.0 + 2*z_val + 3*z_val^2*gs_val + z_val^3*gs_prime_val)) / (hat_q^2)
        term_q_R = (hat_bar_chi * (gs_val + z_val*gs_prime_val) / (κ) ) / (hat_q^2)
        q_temp = term_q_T + term_q_S + term_q_R

        # m
        m_temp = (hat_m / (hat_q)) * (1.0 + z_val*(1 + z_val*gs_val)) 

        # χ
        chi_temp = (1 / (κ*hat_q)) * (1 + z_val*gs_val) 

        # q_bar 
        term_q_bar_T = (hat_chi * (gs_val + z_val*gs_prime_val) / (κ) ) / (hat_q^2)
        term_q_bar_S = (hat_m^2 * (1 + 2*z_val*gs_val + z_val^2*gs_prime_val) ) / (hat_q^2)
        term_q_bar_R = (hat_bar_chi * gs_prime_val / (κ) ) / (hat_q^2)
        q_bar_temp = term_q_bar_T + term_q_bar_S + term_q_bar_R

        # chi_bar
        chi_bar_temp = (1 / (hat_q*κ)) * gs_val  

        # damping
        q = damp * q_temp + (1 - damp) * q
        m = damp * m_temp + (1 - damp) * m
        χ = damp * chi_temp + (1 - damp) * χ
        q_bar = damp * q_bar_temp + (1 - damp) * q_bar
        chi_bar = damp * chi_bar_temp + (1 - damp) * chi_bar

        # convergence check
        diff_vector = [q - q_old, m - m_old, χ - χ_old, q_bar - q_bar_old, chi_bar - chi_bar_old]

        if any(isnan, diff_vector) || any(isinf, diff_vector)
            @warn "NaN or Inf detected in order parameter updates after damping at iter $iter. Aborting."
            return q_old, m_old, χ_old, q_bar_old, chi_bar_old, hat_q, hat_m, hat_chi, hat_bar_q, hat_bar_chi, NaN 
        end
        diff = norm(diff_vector)

        z_val = -(λ + hat_bar_q) / hat_q 
        gS_val = gS_lowrank(z_val, κ, ρ)
        gS_prime_val = gS_prime_lowrank(z_val, κ, ρ, gS_val)

        # m_bar
        m_bar = (hat_m / (hat_q)) * (1.0 + z_val*gS_val) 
        m0 = m_bar / ρ 

        # q0
        gS_prime_tilde_val = gS_prime_tilde(z_val, κ, ρ)
        term_q0_T = ((hat_chi * (gS_val + z_val*gS_prime_val)) / (ρ * κ) ) / (hat_q^2) 
        term_q0_S = ((hat_m^2 * (1 + 2*z_val*gS_val + z_val^2*gS_prime_val)) / (ρ) ) / (hat_q^2)
        term_q0_R = ((hat_bar_chi * gS_prime_tilde_val) / (κ) ) / (hat_q^2)
        q0 = term_q0_T  + term_q0_S + term_q0_R

        if diff < tol
            if decomposed
                return q, m, χ, q_bar, chi_bar, hat_q, hat_m, hat_chi, hat_bar_q, hat_bar_chi, m_bar, q0, m0, term_q_T, term_q_S, term_q_R, term_q_bar_T, term_q_bar_S, term_q_bar_R, term_q0_T, term_q0_S, term_q0_R
            else
                return q, m, χ, q_bar, chi_bar, hat_q, hat_m, hat_chi, hat_bar_q, hat_bar_chi, m_bar, q0, m0
            end
        end
    end


    @warn "Not converged after $max_iter iterations for α=$α, τ=$τ, κ=$κ, λ=$λ"
    if decomposed
        return q, m, χ, q_bar, chi_bar, hat_q, hat_m, hat_chi, hat_bar_q, hat_bar_chi, m_bar, q0, m0, term_q_T, term_q_S, term_q_R, term_q_bar_T, term_q_bar_S, term_q_bar_R, term_q0_T, term_q0_S, term_q0_R
    else
        return q, m, χ, q_bar, chi_bar, hat_q, hat_m, hat_chi, hat_bar_q, hat_bar_chi, m_bar, q0, m0
    end
end



export solve_replica
end # module
