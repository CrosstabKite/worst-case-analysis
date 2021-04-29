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
    - MAP estimate
    - P(click rate less than threshold)
    - E(click rate | click rate less than threshold)
"""

import altair as alt
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st


## Basic setup and app layout
rng = np.random.default_rng(17)
st.set_page_config(layout="wide")  # this needs to be the first Streamlit command called
st.title("Worst-Case Analysis for Feature Rollouts")
st.sidebar.title("Control Panel")
left_col, middle_col, right_col = st.beta_columns(3)


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
st.sidebar.subheader("Prior belief about the click rate")
prior_sessions = st.sidebar.number_input(
    "Number of prior sessions", min_value=1, max_value=None, value=100, step=1
)
prior_click_rate = st.sidebar.slider(
    "Prior click rate", min_value=0.01, max_value=0.5, value=0.1
)

st.sidebar.subheader("Go/no-go decision criteria")
worst_case_threshold = st.sidebar.slider(
    "Worst-case click rate threshold",
    min_value=0.01,
    max_value=0.5,
    value=0.08,
)

worst_case_max_proba = st.sidebar.slider(
    "Max acceptable worst-case probability", min_value=0.0, max_value=1.0, value=0.1
)


## Define the prior
prior_clicks = int(prior_sessions * prior_click_rate)
prior_misses = prior_sessions - prior_clicks
prior = stats.beta(prior_clicks, prior_misses)


## Show key results over time. The index value indicates the data for that day has been
# observed.
results = pd.DataFrame(
    {
        # 'mode': None,
        "mean": None,
        "q05": None,
        "q95": None,
        # 'worst_case_prob': None
    },
    index=range(-1, num_experiment_days),
)

results.loc[-1] = {  # this is the prior, at index -1.
    # 'mode': np.argmax()
    "mean": prior.mean(),
    "q05": prior.ppf(0.05),
    "q95": prior.ppf(0.95),
}

post_clicks = prior_clicks
post_misses = prior_misses

for t in range(num_experiment_days):
    post_clicks = post_clicks + data.loc[t, "clicks"]
    post_misses = post_misses + data.loc[t, "misses"]
    posterior = stats.beta(post_clicks, post_misses)

    results.loc[t] = {
        "mean": posterior.mean(),
        "q05": posterior.ppf(0.05),
        "q95": posterior.ppf(0.95),
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

middle_col.subheader("Observed data")
middle_col.altair_chart(fig, use_container_width=True)


## Draw the posterior zoomed in
xmin = posterior.ppf(0.0001)
xmax = posterior.ppf(0.9999)
distro_grid = np.linspace(xmin, xmax, 300)

posterior_pdf = pd.DataFrame({
    'click_rate': distro_grid,
    'posterior_pdf': [posterior.pdf(x) for x in distro_grid]
})

fig = (
    alt.Chart(posterior_pdf)
    .mark_line(size=4)
    .encode(
        x=alt.X('click_rate', title="Click rate", scale=alt.Scale(domain=[xmin, xmax])),
        y=alt.Y('posterior_pdf', title="Probability density"),
        tooltip=["click_rate", "posterior_pdf"],
    )
)

middle_col.subheader("Zoomed-in posterior belief")
middle_col.altair_chart(fig, use_container_width=True)


## Draw key results over time
results.reset_index(inplace=True)
out = results.melt(id_vars=['index'])

ts_mean = (
    alt.Chart(results)
    .mark_line()
    .encode(
        x="index",
        y="mean",
    )
)

band = (
    alt.Chart(results)
    .mark_area(opacity=0.5)
    .encode(
        x='index',
        y='q05',
        y2='q95'
    )
)

fig = alt.layer(ts_mean, band)

right_col.subheader("Posterior over time")
right_col.altair_chart(fig, use_container_width=True)


## Write key outputs in the control panel
right_col.subheader("Results")

observed_sessions = data['sessions'].sum()
observed_clicks = data['clicks'].sum()
observed_click_rate = observed_clicks / observed_sessions

right_col.markdown(f"**Observed sessions:** {observed_sessions}")
right_col.markdown(f"**Observed clicks:** {observed_clicks}")
right_col.markdown(f"**Observed click rate:** {observed_click_rate}")
right_col.markdown(f"**Most likely posterior click rate:** {17}")
right_col.markdown(f"**Mean posterior click rate:** {results.loc[6]['mean']}")
right_col.markdown(f"**90% credible region for click rate:** [{results.loc[6]['q05']}, {results.loc[6]['q95']}]")
right_col.markdown(f"**Probability click rate less than business threshold:** {13}")
