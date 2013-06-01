abstract GenericDiscriminantAnalysis <: ContinuousMultivariateDistribution

immutable GaussianDiscriminantAnalysis <: GenericDiscriminantAnalysis
	mu::Matrix{Float64}
	sigma::Array{Float64}
	p::Vector{Float64}
	normals::Vector{MultivariateNormal}
	function GaussianDiscriminantAnalysis(mu, sigma, pi)
		p, k = size(mu)
		if size(sigma) != (p, p, k)
			error("Dimensions of mu and sigma")
		end
		if length(pi) != k
			error("Length of pi")
		end
		normals = Array(MultivariateNormal, k)
		for i in 1:k
			normals[i] = MultivariateNormal(mu[:, i], sigma[:, :, i])
		end
		new(mu, sigma, pi, normals)
	end
end

typealias QuadraticDiscriminantAnalysis GaussianDiscriminantAnalysis
typealias GDA GaussianDiscriminantAnalysis
typealias QDA GaussianDiscriminantAnalysis

immutable LinearDiscriminantAnalysis <: GenericDiscriminantAnalysis
	mu::Matrix{Float64}
	sigma::Matrix{Float64}
	p::Vector{Float64}
	normals::Vector{MultivariateNormal}
	function LinearDiscriminantAnalysis(mu, sigma, pi)
		p, k = size(mu)
		if size(sigma) != (p, p)
			error("Dimensions of mu and sigma")
		end
		if length(pi) != k
			error("Length of pi")
		end
		normals = Array(MultivariateNormal, k)
		for i in 1:k
			normals[i] = MultivariateNormal(mu[:, i], sigma)
		end
		new(mu, sigma, pi, normals)
	end
end

typealias LDA LinearDiscriminantAnalysis

immutable RegularizedLinearDiscriminantAnalysis <: GenericDiscriminantAnalysis
	mu::Matrix{Float64}
	sigma::Matrix{Float64}
	p::Vector{Float64}
	normals::Vector{MultivariateNormal}
	lambda::Float64
	function RegularizedLinearDiscriminantAnalysis(mu, sigma, pi, lambda)
		p, k = size(mu)
		if size(sigma) != (p, p)
			error("Dimensions of mu and sigma")
		end
		if length(pi) != k
			error("Length of pi")
		end
		normals = Array(MultivariateNormal, k)
		for i in 1:k
			normals[i] = MultivariateNormal(mu[:, i], sigma)
		end
		new(mu, sigma, pi, normals, lambda)
	end
end

typealias rLDA RegularizedLinearDiscriminantAnalysis

# TODO: Debate storing only diagonals for efficiency
immutable DiagonalLinearDiscriminantAnalysis <: GenericDiscriminantAnalysis
	mu::Matrix{Float64}
	sigma::Matrix{Float64}
	p::Vector{Float64}
	normals::Vector{MultivariateNormal}
	function DiagonalLinearDiscriminantAnalysis(mu, sigma, pi)
		p, k = size(mu)
		if size(sigma) != (p, p)
			error("Dimensions of mu and sigma")
		end
		if length(pi) != k
			error("Length of pi")
		end
		normals = Array(MultivariateNormal, k)
		for i in 1:k
			normals[i] = MultivariateNormal(mu[:, i], sigma)
		end
		new(mu, sigma, pi, normals)
	end
end

typealias dLDA DiagonalLinearDiscriminantAnalysis

function Base.show(io::IO, a::GenericDiscriminantAnalysis)
	k = size(a.mu, 2)
	@printf io "Discriminant Analysis Results with %d Groups\n" k
	for i in 1:k
		@printf io " * %d: %s\n" i a.mu[:, i]
	end
end

function Base.mean(d::GenericDiscriminantAnalysis)
	d.mu * d.p
end
