module DiscriminantAnalysis
	using Distributions
	importall Distributions

	export GDA, QDA, LDA, rLDA, dLDA

	export GaussianDiscriminantAnalysis
	export QuadraticDiscriminantAnalysis
	export LinearDiscriminantAnalysis
	export RegularizedLinearDiscriminantAnalysis
	export DiagonalLinearDiscriminantAnalysis

	# Should be in Distributions
	export predict

	include("utils.jl")
	include("types.jl")
	include("fit.jl")
	include("rand.jl")
	include("pdf.jl")
end
