"""
Streamlit app for Bayesian worst-case analysis of a new feature rollout.

Based on this article
https://www.crosstab.io/articles/confidence-interval-interpretation, the scenario is
that we're doing a staged rollout of a new feature on our company website, and we need
to decide if we should proceed to the next stage. In this scenario, we want a
*non-inferiority* analysis, not a traditional A/B test superiority analysis. We also
want to use a Bayesian approach, so that our conclusions about the true click-rate have
the interpretation that matches most decision-makers' intuition.

TODO
- Add this volume histogram to the data plot
- Add stub controls for the prior to the sidebar
- Add worst-case distribution in the bottom right plot
- Add numeric outputs somewhere
"""

import altair as alt
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st


### Numeric computation
### -------------------

## Numeric set up
rng = np.random.default_rng(17)
xgrid = np.linspace(0, 1, 200)  # cache this


## Define the prior
prior_clicks = 100
prior_misses = 900
prior = stats.beta(prior_clicks, prior_misses)

prior_pdf = pd.DataFrame({
    'click_rate': xgrid,
    'prior_pdf': [prior.pdf(x) for x in xgrid]
})


## Simulate data - cache this
click_rate = 0.08
mean_sessions_per_day = 1000
num_days = 7

sessions = stats.poisson.rvs(mu=mean_sessions_per_day, size=num_days, random_state=rng)
clicks = np.array([stats.binom.rvs(n=n, p=click_rate, size=1, random_state=rng)[0] for n in sessions])
misses = sessions - clicks

data = pd.DataFrame({
    'day': range(num_days),
    'sessions': sessions,
    'clicks': clicks,
    'misses': misses,
    'click_rate': clicks / sessions
})


## Compute the posterior
post_clicks = sum(clicks) + prior_clicks
post_misses = sum(misses) + prior_misses

posterior = stats.beta(post_clicks, post_misses)

posterior_pdf = pd.DataFrame({
    'click_rate': xgrid,
    'posterior_pdf': [posterior.pdf(x) for x in xgrid]
})


### Construct the app
### -----------------

## App set up
st.set_page_config(layout="wide")
st.title("Worst-Case Analysis for Feature Rollouts")
st.sidebar.title("Control Panel")
left_col, right_col = st.beta_columns(2)

## Draw the prior
fig = (
    alt.Chart(prior_pdf)
    .mark_line()
    .encode(
        x='click_rate',
        y='prior_pdf'
    )
)

left_col.altair_chart(fig, use_container_width=True)

## Draw the posterior
fig = (
    alt.Chart(posterior_pdf)
    .mark_line()
    .encode(
        x='click_rate',
        y='posterior_pdf'
    )
)

left_col.altair_chart(fig, use_container_width=True)

## Draw the data
fig = (
    alt.Chart(data)
    .mark_line()
    .encode(
        x="day",
        y="click_rate"
    )
)

right_col.altair_chart(fig, use_container_width=True)
