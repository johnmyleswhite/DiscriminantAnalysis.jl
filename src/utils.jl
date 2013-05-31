# Want to estimate means and covariances with any copying
function estimate_mean(X::Matrix)
	vec(mean(X, 2))
end

function estimate_covariance(X::Matrix)
	cov(X')
end
