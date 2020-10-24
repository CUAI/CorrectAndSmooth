using LinearAlgebra
using LinearMaps
using MAT
using SparseArrays
using Arpack

using PyCall, SparseArrays

function scipyCSC_to_julia(A)
    m, n = A.shape
    colPtr = Int[i+1 for i in PyArray(A."indptr")]
    rowVal = Int[i+1 for i in PyArray(A."indices")]
    nzVal = Vector{Float64}(PyArray(A."data"))
    B = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)
    return PyCall.pyjlwrap_new(B)
end

function read_arxiv(file::String)
    I = Int64[]
    J = Int64[]
    open(file) do f
        for line in eachline(f)
            if line[1] == '#'; continue; end
            data = split(line, ",")
            push!(I, parse(Int64, data[1]))
            push!(J, parse(Int64, data[2]))
        end
    end
    I .+= 1
    J .+= 1    
    n = max(maximum(I), maximum(J))
    A = sparse(I, J, 1, n, n)
    A = max.(A, A')
    A = min.(A, 1)
    return A
end


function main(PyA, k::Int64)
    m, n = PyA.shape
    colPtr = Int[i+1 for i in PyArray(PyA."indptr")]
    rowVal = Int[i+1 for i in PyArray(PyA."indices")]
    nzVal = Vector{Float64}(PyArray(PyA."data"))
    A = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)
    d = vec(sum(A, dims=2))
    τ = sum(d) / length(d)
    N = size(A)[1]

    # normalized regularized laplacian
    D = Diagonal(1.0 ./ sqrt.(d .+ τ))
    Aop = LinearMap{Float64}(X -> A * X .+ (τ / N) * sum(X), N, N, isposdef=true, issymmetric=true)
    NRL = I + D * Aop * D

    (Λ, V) = eigs(NRL, nev=k, tol=1e-6, ncv=2*k+1, which=:LM)

    # axis rotation (not necessary, but could be helpful)
    piv = qr(V', Val(true)).jpvt[1:k]
    piv_svd = svd(V[piv,:]', full=false)
    SCDM_V = V * (piv_svd.U * piv_svd.Vt)

    # save

    return SCDM_V
end

#A = read_arxiv(ARGS[1])
#embed = main(A, 128)
#matwrite("$(ARGS[2])_spectral_embedding.mat", Dict("V" => embed), compress=true)

