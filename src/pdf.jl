# pdf, logpdf, loglikelihood, ...
function pdf(d::GenericDiscriminantAnalysis, x::Vector)
	p, k = size(d.mu)
	prob = 1.0
	for i in 1:k
		prob *= pdf(d.normals[i], x) * d.p[i]
	end
	return prob
end

function pdf(d::GenericDiscriminantAnalysis, x::Vector, y::Real)
	return pdf(d.normals[y], x) * d.p[y]
end

# Produce pdf for Matrix, Vector

function logpdf(d::GenericDiscriminantAnalysis, x::Vector)
	p, k = size(d.mu)
	lp = 0.0
	for i in 1:k
		lp += logpdf(d.normals[i], x) + log(d.p[i])
	end
	return lp
end

function logpdf(d::GenericDiscriminantAnalysis, x::Vector, y::Real)
	return logpdf(d.normals[y], x) + log(d.p[y])
end

function predict(d::GenericDiscriminantAnalysis, x::Vector)
	p, k = size(d.mu)
	probs = Array(Float64, k)
	for i in 1:k
		probs[i] = logpdf(d, x, i)
	end
	return indmax(probs)
end

function predict(d::GenericDiscriminantAnalysis, X::Matrix)
	p, k = size(d.mu)
	p, n = size(X)
	# probs = Array(Float64, k)
	preds = Array(Int, n)
	for i in 1:n
		preds[i] = predict(d, X[:, i])
	end
	return preds
end

function loglikelihood(d::GenericDiscriminantAnalysis, X::Matrix, y::Vector)
	ll = 0.0
	p, n = size(X)
	for i in 1:n
		ll += logpdf(d, X[:, i], y[i])
	end
	return ll
end

function loglikelihood(d::GenericDiscriminantAnalysis, X::Matrix)
	ll = 0.0
	p, n = size(X)
	for i in 1:n
		ll += logpdf(d, X[:, i])
	end
	return ll
end
