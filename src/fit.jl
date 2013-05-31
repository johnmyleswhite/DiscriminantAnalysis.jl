# GDA: Estimate means and full covariance matrices by MLE
# for all classes
function GDA(X::Matrix, y::Vector)
	p, n = size(X)
	@assert length(y) == n
	k = max(y) # Assume all classes exist
	mu = Array(Float64, p, k)
	sigma = Array(Float64, p, p, k)
	for class in 1:k
		indices = find(y .== class)
		mu[:, class] = estimate_mean(X[:, indices])
		sigma[:, :, class] = estimate_covariance(X[:, indices])
	end
	return mu, sigma
end

# LDA: Estimate means by MLE for all classes and a shared full
# covariance matrix by MLE ignoring classes
function LDA(X::Matrix, y::Vector)
	p, n = size(X)
	@assert length(y) == n
	k = max(y) # Assume all classes exist
	mu = Array(Float64, p, k)
	sigma = Array(Float64, p, p)
	for class in 1:k
		indices = find(y .== class)
		mu[:, class] = estimate_mean(X[:, indices])
	end
	sigma = estimate_covariance(X)
	return mu, sigma
end

# RDA: Estimate means by MLE for all classes and a shared full
# covariance matrix by MLE ignoring classes. Then take convex
# combination (with parameter lambda) combining S and diagm(diag(S)).
function RDA(X::Matrix, y::Vector, lambda::Real = 0.0)
	@assert 0.0 <= lambda <= 1.0
	p, n = size(X)
	@assert length(y) == n
	k = max(y) # Assume all classes exist
	mu = Array(Float64, p, k)
	sigma = Array(Float64, p, p)
	for class in 1:k
		indices = find(y .== class)
		mu[:, class] = estimate_mean(X[:, indices])
	end
	sigma = estimate_covariance(X)
	sigma = lambda * diagm(diag(sigma)) + (1.0 - lambda) * sigma
	return mu, sigma
end

# Diagonal LDA: Estimate means by MLE for all classes and a shared
# diagonal covariance matrix by MLE ignoring classes. Equivalent
# to Gaussian Naive Bayes.
function dLDA(X::Matrix, y::Vector)
	p, n = size(X)
	@assert length(y) == n
	k = max(y) # Assume all classes exist
	mu = Array(Float64, p, k)
	sigma = Array(Float64, p, p)
	for class in 1:k
		indices = find(y .== class)
		mu[:, class] = estimate_mean(X[:, indices])
	end
	sigma = estimate_covariance(X)
	sigma = diagm(diag(sigma))
	return mu, sigma
end
