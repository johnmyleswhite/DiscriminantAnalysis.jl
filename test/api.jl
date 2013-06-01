using Distributions
using DiscriminantAnalysis

d1 = MultivariateNormal([0.0, 0.0], [2.0 1.0;
	                                 1.0 2.0])

d2 = MultivariateNormal([6.0, 6.0], [3.0 0.0;
	                                   0.0 3.0])

X = hcat(rand(d1, 100), rand(d2, 100))

y = vec(repmat([1, 2], 1, 100)')

fit(GaussianDiscriminantAnalysis, X, y)
fit(GDA, X, y)

fit(QuadraticDiscriminantAnalysis, X, y)
fit(QDA, X, y)

fit(LinearDiscriminantAnalysis, X, y)
fit(LDA, X, y)

fit(RegularizedLinearDiscriminantAnalysis, X, y, 0.0)
fit(rLDA, X, y, 0.0)

fit(RegularizedLinearDiscriminantAnalysis, X, y, 0.5)
fit(rLDA, X, y, 0.5)

fit(RegularizedLinearDiscriminantAnalysis, X, y, 1.0)
fit(rLDA, X, y, 1.0)

fit(DiagonalLinearDiscriminantAnalysis, X, y)
fit(dLDA, X, y)

d = fit(GDA, X, y)
Xprime, yprime = rand(d, 10000)
fit(GDA, Xprime, yprime)

d = fit(LDA, X, y)
Xprime, yprime = rand(d, 10000)
fit(LDA, Xprime, yprime)

loglikelihood(fit(QDA, X, y), X, y)
loglikelihood(fit(LDA, X, y), X, y)
loglikelihood(fit(rLDA, X, y, 0.5), X, y)
loglikelihood(fit(dLDA, X, y), X, y)

mean(predict(fit(QDA, X, y), X) .== y)
mean(predict(fit(LDA, X, y), X) .== y)
mean(predict(fit(rLDA, X, y, 0.5), X) .== y)
mean(predict(fit(dLDA, X, y), X) .== y)
