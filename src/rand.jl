# rand()

function Distributions.rand(d::GenericDiscriminantAnalysis)
	c = rand(Categorical(d.p))
	return rand(d.normals[c]), c
end

function Distributions.rand(d::GenericDiscriminantAnalysis, n::Integer)
	p, k = size(d.mu)
	X = Array(Float64, p, n)
	y = Array(Int, n)
	for i in 1:n
		c = rand(Categorical(d.p))
		y[i] = c
		X[:, i] = rand(d.normals[c])
	end
	return X, y
end
