# GDA: Estimate means and full covariance matrices by MLE
# for all classes
function fitGDA(X::Matrix, y::Vector)
	p, n = size(X)
	@assert length(y) == n
	k = max(y) # Assume all classes exist
	mu = Array(Float64, p, k)
	sigma = Array(Float64, p, p, k)
	pi = Array(Float64, k)
	for class in 1:k
		indices = find(y .== class)
		pi[class] = length(indices) / n
		mu[:, class] = estimate_mean(X[:, indices])
		sigma[:, :, class] = estimate_covariance(X[:, indices])
	end
	return GaussianDiscriminantAnalysis(mu, sigma, pi)
end

function Distributions.fit(::Type{GaussianDiscriminantAnalysis},
	                       X::Matrix,
	                       y::Vector)
	fitGDA(X, y)
end

# LDA: Estimate means by MLE for all classes and a shared full
# covariance matrix by MLE ignoring classes
function fitLDA(X::Matrix, y::Vector)
	p, n = size(X)
	@assert length(y) == n
	k = max(y) # Assume all classes exist
	mu = Array(Float64, p, k)
	sigma = Array(Float64, p, p)
	pi = Array(Float64, k)
	for class in 1:k
		indices = find(y .== class)
		pi[class] = length(indices) / n
		mu[:, class] = estimate_mean(X[:, indices])
	end
	Xcentered = copy(X)
	for i in 1:n
		Xcentered[:, i] = X[:, i] - mu[:, y[i]]
	end
	sigma = estimate_covariance(Xcentered)
	return LinearDiscriminantAnalysis(mu, sigma, pi)
end

function Distributions.fit(::Type{LinearDiscriminantAnalysis},
	                       X::Matrix,
	                       y::Vector)
	fitLDA(X, y)
end

# RDA: Estimate means by MLE for all classes and a shared full
# covariance matrix by MLE ignoring classes. Then take convex
# combination (with parameter lambda) combining S and diagm(diag(S)).
function fitrLDA(X::Matrix, y::Vector, lambda::Real = 0.0)
	@assert 0.0 <= lambda <= 1.0
	p, n = size(X)
	@assert length(y) == n
	k = max(y) # Assume all classes exist
	mu = Array(Float64, p, k)
	sigma = Array(Float64, p, p)
	pi = Array(Float64, k)
	for class in 1:k
		indices = find(y .== class)
		pi[class] = length(indices) / n
		mu[:, class] = estimate_mean(X[:, indices])
	end
	Xcentered = copy(X)
	for i in 1:n
		Xcentered[:, i] = X[:, i] - mu[:, y[i]]
	end
	sigma = estimate_covariance(Xcentered)
	sigma = lambda * diagm(diag(sigma)) + (1.0 - lambda) * sigma
	return RegularizedLinearDiscriminantAnalysis(mu, sigma, pi, lambda)
end

function Distributions.fit(::Type{RegularizedLinearDiscriminantAnalysis},
	                       X::Matrix,
	                       y::Vector,
	                       lambda)
	fitrLDA(X, y, lambda)
end

# Diagonal LDA: Estimate means by MLE for all classes and a shared
# diagonal covariance matrix by MLE ignoring classes. Equivalent
# to Gaussian Naive Bayes.
function fitdLDA(X::Matrix, y::Vector)
	p, n = size(X)
	@assert length(y) == n
	k = max(y) # Assume all classes exist
	mu = Array(Float64, p, k)
	sigma = Array(Float64, p, p)
	pi = Array(Float64, k)
	for class in 1:k
		indices = find(y .== class)
		pi[class] = length(indices) / n
		mu[:, class] = estimate_mean(X[:, indices])
	end
	Xcentered = copy(X)
	for i in 1:n
		Xcentered[:, i] = X[:, i] - mu[:, y[i]]
	end
	sigma = estimate_covariance(Xcentered)
	sigma = diagm(diag(sigma))
	return DiagonalLinearDiscriminantAnalysis(mu, sigma, pi)
end

function Distributions.fit(::Type{DiagonalLinearDiscriminantAnalysis},
	                       X::Matrix,
	                       y::Vector)
	fitdLDA(X, y)
end
