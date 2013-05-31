DiscriminantAnalysis.jl
=======================

# Usage

Here is the basic API as it stands now:

	using Distributions
	using DiscriminantAnalysis

	d1 = MultivariateNormal([0.0, 0.0], [2.0 1.0;
		                                 1.0 2.0])

	d2 = MultivariateNormal([10.0, 10.0], [3.0 0.5;
		                                   0.5 3.0])

	X = hcat(rand(d1, 100), rand(d2, 100))

	y = vec(repmat([1, 2], 1, 100)')

	GDA(X, y)
	LDA(X, y)
	RDA(X, y, 0.0)
	RDA(X, y, 0.5)
	RDA(X, y, 1.0)
	dLDA(X, y)

# To Do

* Add nearest shrunken centroids
* Add methods to handle rank deficient matrices
* Revise code for performance
