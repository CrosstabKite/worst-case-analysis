"""
Play with a Bayesian "worst-case" analysis of a click-through rate experiment.

TODO
- How compute posterior quantities?
- How should non-experts specify the prior?
- What's the meaning of the beta params again? Is `b` prior misses or the total?

NOTE
- One main point is how robust the results are to the prior.
"""

import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

rng = np.random.default_rng(17)


## User controls, i.e. parameters
prior_clicks = 100
prior_misses = 900


## The prior - beta distro
xgrid = np.linspace(0, 1, 200)
prior = [stats.beta.pdf(x, prior_clicks, prior_misses) for x in xgrid]

fig = go.Figure(go.Scatter(
    x=xgrid, y=prior
))


## The data - binomial, maybe a time series to imagine drawing more data?
# - This would be observed in reality, but here we need to simulate
click_rate = 0.08
mean_sessions_per_day = 1000
num_days = 7

sessions = stats.poisson.rvs(mu=mean_sessions_per_day, size=num_days, random_state=rng)
clicks = np.array([stats.binom.rvs(n=n, p=click_rate, size=1, random_state=rng)[0] for n in sessions])
misses = sessions - clicks


## The posterior
# - How to toggle between quantities of interest?
post_clicks = sum(clicks) + prior_clicks
post_misses = sum(misses) + prior_misses

posterior = stats.beta(post_clicks, post_misses)

post_pdf = [posterior.pdf(x) for x in xgrid]

fig.add_trace(go.Scatter(
    x=xgrid, y=post_pdf
))
fig.show()


## Numeric summaries and final decisions

# 1. Credible interval
ci = posterior.interval(0.95)  # not exactly sure how this works

q025 = posterior.ppf(0.025)
q975 = posterior.ppf(0.975)

# 2. P(\theta < threshold)
worst_case_threshold = 0.08
worst_case_prob = posterior.cdf(worst_case_threshold)

# 3. Given that theta is below some threshold, what's the distribution?
# 4. What else?

