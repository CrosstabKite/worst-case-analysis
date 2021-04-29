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
- Add numeric outputs somewhere (maybe the control panel?)
- Draw the worst-case threshold on the plot
- Draw the worst-case probability as filled area on each of the plots
- Draw the worst-case probability over time on the time series plot
- Fill in the area for the credible region over time plot

- Does it really add anything to have the user iterate through the days, since I have
  the key outcomes over time?

- Add text to explain how to provide the prior numbers
- Increase font sizes of the axis labels and ticks
- Format the tooltip
- Tooltips for observed data
- Get the sessions volume histogram to line up properly (or think of another way to do
  it, i.e. with bubble sizes)
- Think about whether to guide to Jeffreys priors/maximally noninformative priors
- In the article, make a section for further reading

NOTE
- Possible additional plots
    - Zoom on the posterior
    - Distribution if the worst-case happens
    - Posterior quantities (mean, credible interval) by experiment day

- Additional feature brainstorm
    - Compare two priors

- Step through days of the experiment, or backward
- Animate stepping through days of the experiment

- Summary stats
    - total amount of data observed so far, sessions and clicks
    - MAP estimate
    - 95% CR
    - P(click rate less than threshold)
    - E(click rate | click rate less than threshold)
"""

import altair as alt
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st


## Basic setutp
rng = np.random.default_rng(17)
st.set_page_config(layout="wide")
st.title("Worst-Case Analysis for Feature Rollouts")
st.sidebar.title("Control Panel")
left_col, right_col = st.beta_columns(2)


## Simulate data and the distribution domain
@st.cache
def generate_data(click_rate, avg_daily_sessions, num_days):
    """Simulate session and click counts for some number of days."""

    sessions = stats.poisson.rvs(mu=avg_daily_sessions, size=num_days, random_state=rng)
    clicks = np.array(
        [
            stats.binom.rvs(n=n, p=click_rate, size=1, random_state=rng)[0]
            for n in sessions
        ]
    )
    misses = sessions - clicks

    data = pd.DataFrame(
        {
            "day": range(num_days),
            "sessions": sessions,
            "clicks": clicks,
            "misses": misses,
            "click_rate": clicks / sessions,
        }
    )

    return data


num_experiment_days = 14
data = generate_data(
    click_rate=0.08, avg_daily_sessions=500, num_days=num_experiment_days
)


## User inputs on the control panel
prior_sessions = st.sidebar.number_input(
    "Number of prior sessions", min_value=1, max_value=None, value=100, step=1
)
prior_click_rate = st.sidebar.slider(
    "Prior click rate", min_value=0.01, max_value=0.5, value=0.1
)
worst_case_threshold = st.sidebar.slider(
    "Worst-case click rate threshold",
    min_value=0.01,
    max_value=0.5,
    value=0.08,
)

experiment_day = st.sidebar.slider("Experiment day", min_value=0, max_value=num_experiment_days, value=7)


## Define the prior
prior_clicks = int(prior_sessions * prior_click_rate)
prior_misses = prior_sessions - prior_clicks
prior = stats.beta(prior_clicks, prior_misses)


## Compute the posterior using all the data seen so far
data = data.query("day < @experiment_day")


## Show key statistics over time
results = pd.DataFrame({
    # 'mode': None,
    'mean': None,
    'q05': None,
    'q95': None,
    # 'worst_case_prob': None
}, index=range(-1, experiment_day))

# The prior
results.loc[-1] = {
    # 'mode': np.argmax()
    'mean': prior.mean(),
    'q05': prior.ppf(0.05),
    'q95': prior.ppf(0.95)
}

post_clicks = prior_clicks
post_misses = prior_misses

for t in range(experiment_day):
    post_clicks = post_clicks + data.loc[t, 'clicks']
    post_misses = post_misses + data.loc[t, 'misses']
    posterior = stats.beta(post_clicks, post_misses)

    results.loc[t] = {
        'mean': posterior.mean(),
        'q05': posterior.ppf(0.05),
        'q95': posterior.ppf(0.95)
    }
    

## Get the max useful click rate value to show in the distribution plots
xmax = max(prior.ppf(0.99999), posterior.ppf(0.99999))
distro_grid = np.linspace(0, xmax, 300)

 
## Draw the prior
prior_pdf = pd.DataFrame(
    {"click_rate": distro_grid, "prior_pdf": [prior.pdf(x) for x in distro_grid]}
)

fig = (
    alt.Chart(prior_pdf)
    .mark_line(size=4)
    .encode(
        x=alt.X("click_rate", title="Click rate"),
        y=alt.Y("prior_pdf", title="Probability density"),
        tooltip=["click_rate", "prior_pdf"],
    )
)

left_col.subheader("Prior belief about the click rate")
left_col.altair_chart(fig, use_container_width=True)


## Draw the final posterior
posterior_pdf = pd.DataFrame({
    'click_rate': distro_grid,
    'posterior_pdf': [posterior.pdf(x) for x in distro_grid]
})

fig = (
    alt.Chart(posterior_pdf)
    .mark_line(size=4)
    .encode(
        x=alt.X('click_rate', title="Click rate", scale=alt.Scale(domain=[0, xmax])),
        y=alt.Y('posterior_pdf', title="Probability density"),
        tooltip=["click_rate", "posterior_pdf"],
    )
)

left_col.subheader("Updated posterior belief about the click rate")
left_col.altair_chart(fig, use_container_width=True)


## Draw the data
base = alt.Chart(data).encode(
    alt.X("day", title="Experiment day", scale=alt.Scale(domain=[0, num_experiment_days]))
)

volume_fig = base.mark_bar(color="orange").encode(
    y=alt.Y('sessions', axis=alt.Axis(title="Number of sessions", titleColor="orange"))
)

rate_fig = base.mark_line(size=4, color="blue").encode(
    y=alt.Y("click_rate", axis=alt.Axis(title="Click rate", titleColor="blue"))
)

fig = alt.layer(volume_fig, rate_fig).resolve_scale(y="independent")

right_col.subheader("Observed data")
right_col.altair_chart(fig, use_container_width=True)


## Draw key results over time
results.reset_index(inplace=True)
out = results.melt(id_vars=['index'])

fig = (
    alt.Chart(out)
    .mark_line()
    .encode(
        x="index",
        y="value",
        color="variable"
    )
)

right_col.subheader("Posterior quantities of interest")
right_col.altair_chart(fig, use_container_width=True)
