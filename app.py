"""
Streamlit app for Bayesian worst-case analysis of a new feature rollout.

Based on this article
https://www.crosstab.io/articles/confidence-interval-interpretation, the scenario is
that we're doing a staged rollout of a new feature on our company website, and we need
to decide if we should proceed to the next stage. In this scenario, we want a
*non-inferiority* analysis, not a traditional A/B test superiority analysis. We also
want to use a Bayesian approach, so that our conclusions about the true click-rate have
the interpretation that matches most decision-makers' intuition.

NOTE
- Figures are saved in SVG format if this script is run in a REPL instead of Streamlit.
  It would be better to save directly to PNG, but I can't figure out how to control the
  resolution of the Altair figure saver. This way, I can use Inkscape to convert to PNG
  with the desired resolution.
"""

import altair as alt
import copy
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st


alt.renderers.set_embed_options(scaleFactor=2)


## Basic setup and app layout
st.set_page_config(layout="wide")  # this needs to be the first Streamlit command called
st.title("Worst-Case Analysis for Feature Rollouts")
st.sidebar.title("Control Panel")
left_col, middle_col, right_col = st.beta_columns(3)

tick_size = 12
axis_title_size = 16

## Simulate data and the distribution domain
@st.cache
def generate_data(click_rate, avg_daily_sessions, num_days):
    """Simulate session and click counts for some number of days."""

    sessions = stats.poisson.rvs(mu=avg_daily_sessions, size=num_days, random_state=rng)
    clicks = stats.binom.rvs(n=sessions, p=click_rate, size=num_days, random_state=rng)

    data = pd.DataFrame(
        {
            "day": range(num_days),
            "sessions": sessions,
            "clicks": clicks,
            "misses": sessions - clicks,
            "click_rate": clicks / sessions,
        }
    )

    return data


rng = np.random.default_rng(2)
num_days = 14
data = generate_data(click_rate=0.085, avg_daily_sessions=500, num_days=num_days)


## User inputs on the control panel
st.sidebar.subheader("Prior belief about the click rate")
prior_sessions = st.sidebar.number_input(
    "Number of prior sessions", min_value=1, max_value=None, value=100, step=1
)
prior_click_rate = st.sidebar.slider(
    "Prior click rate", min_value=0.01, max_value=0.5, value=0.1, step=0.005
)

st.sidebar.subheader("Decision criteria")
worst_case_threshold = st.sidebar.slider(
    "Worst-case click rate threshold",
    min_value=0.01,
    max_value=0.5,
    value=0.08,
    step=0.005,
)

worst_case_max_proba = st.sidebar.slider(
    "Max acceptable worst-case probability",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01,
)


## Define the prior
prior_clicks = int(prior_sessions * prior_click_rate)
prior_misses = prior_sessions - prior_clicks

prior = stats.beta(prior_clicks, prior_misses)


## Show key results over time. The index value indicates the data for that day has been
# observed.
results = pd.DataFrame(
    {
        "mean": [prior.mean()],
        "p10": [prior.ppf(0.1)],
        "p90": [prior.ppf(0.9)],
    },
    index=[-1],
)

posterior = copy.copy(prior)
assert(id(posterior) != id(prior))

for t in range(num_days):

    # This is the key posterior update logic, from the beta-binomial conjugate family.
    posterior = stats.beta(
        posterior.args[0] + data.loc[t, "clicks"],
        posterior.args[1] + data.loc[t, "misses"]
    )

    results.at[t] = {
        "mean": posterior.mean(),
        "p10": posterior.ppf(0.1),
        "p90": posterior.ppf(0.9),
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
        tooltip=[
            alt.Tooltip("click_rate", title="Click rate", format=".3f"),
            alt.Tooltip("prior_pdf", title="Probability density", format=".3f"),
        ],
    )
)

threshold_rule = (
    alt.Chart(pd.DataFrame({"x": [worst_case_threshold]}))
    .mark_rule(size=2, color="red")
    .encode(x="x")
)

worst_case_prior_pdf = prior_pdf[prior_pdf["click_rate"] < worst_case_threshold]

worst_case_area = (
    alt.Chart(worst_case_prior_pdf)
    .mark_area(opacity=0.5)
    .encode(x="click_rate", y="prior_pdf")
)

fig = alt.layer(fig, threshold_rule, worst_case_area).configure_axis(
    labelFontSize=tick_size, titleFontSize=axis_title_size
)

if st.util.env_util.is_repl():
    fig.save("worst_case_prior.svg")

left_col.subheader("Prior belief about the click rate")
left_col.altair_chart(fig, use_container_width=True)


## Draw the final posterior
posterior_pdf = pd.DataFrame(
    {
        "click_rate": distro_grid,
        "posterior_pdf": [posterior.pdf(x) for x in distro_grid],
    }
)

fig = (
    alt.Chart(posterior_pdf)
    .mark_line(size=4)
    .encode(
        x=alt.X("click_rate", title="Click rate", scale=alt.Scale(domain=[0, xmax])),
        y=alt.Y("posterior_pdf", title="Probability density"),
        tooltip=["click_rate", "posterior_pdf"],
    )
)

threshold_rule = (
    alt.Chart(pd.DataFrame({"x": [worst_case_threshold]}))
    .mark_rule(size=2, color="red")
    .encode(x="x")
)

fig = alt.layer(fig, threshold_rule).configure_axis(
    labelFontSize=tick_size, titleFontSize=axis_title_size
)

left_col.subheader("Posterior belief about the click rate")
left_col.altair_chart(fig, use_container_width=True)


## Draw the data
base = alt.Chart(data).encode(alt.X("day", title="Experiment day"))

volume_fig = base.mark_bar(color="#ffbb78", size=12).encode(
    y=alt.Y("sessions", axis=alt.Axis(title="Number of sessions", titleColor="#ff7f0e"))
)

rate_fig = base.mark_line(size=4).encode(
    y=alt.Y("click_rate", axis=alt.Axis(title="Click rate", titleColor="#1f77b4"))
)

fig = (
    alt.layer(volume_fig, rate_fig)
    .resolve_scale(y="independent")
    .configure_axis(labelFontSize=tick_size, titleFontSize=axis_title_size)
)

if st.util.env_util.is_repl():
    fig.save("worst_case_data.svg")

middle_col.subheader("Observed data")
middle_col.altair_chart(fig, use_container_width=True)


## Draw the posterior zoomed in
xmin = posterior.ppf(0.0001)
xmax = posterior.ppf(0.9999)
distro_grid = np.linspace(xmin, xmax, 300)

posterior_pdf = pd.DataFrame(
    {
        "click_rate": distro_grid,
        "posterior_pdf": [posterior.pdf(x) for x in distro_grid],
    }
)

distro_fig = (
    alt.Chart(posterior_pdf)
    .mark_line(size=4)
    .encode(
        x=alt.X("click_rate", title="Click rate", scale=alt.Scale(domain=[xmin, xmax])),
        y=alt.Y("posterior_pdf", title="Probability density"),
        tooltip=["click_rate", "posterior_pdf"],
    )
)

threshold_rule = (
    alt.Chart(pd.DataFrame({"x": [worst_case_threshold]}))
    .mark_rule(size=2, color="red", clip=True)
    .encode(x="x")
)

worst_case_post_pdf = posterior_pdf[posterior_pdf["click_rate"] < worst_case_threshold]

worst_case_area = (
    alt.Chart(worst_case_post_pdf)
    .mark_area(opacity=0.5)
    .encode(x="click_rate", y="posterior_pdf")
)

fig = alt.layer(distro_fig, threshold_rule, worst_case_area).configure_axis(
    labelFontSize=tick_size, titleFontSize=axis_title_size
)

if st.util.env_util.is_repl():
    fig.save("worst_case_posterior.svg")

middle_col.subheader("Zoomed-in posterior belief")
middle_col.altair_chart(fig, use_container_width=True)


## Draw key results over time
results.reset_index(inplace=True)
out = results.melt(id_vars=["index"])

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
        x=alt.X("index", title="Experiment day"),
        y=alt.Y("p10", title="Click rate"),
        y2="p90",
    )
)

threshold_rule = (
    alt.Chart(pd.DataFrame({"y": [worst_case_threshold]}))
    .mark_rule(size=2, color="red")
    .encode(y="y")
)

fig = alt.layer(ts_mean, band, threshold_rule).configure_axis(
    labelFontSize=tick_size, titleFontSize=axis_title_size
)

if st.util.env_util.is_repl():
    fig.save("worst_case_posterior_ts.svg")

right_col.subheader("Posterior over time")
right_col.altair_chart(fig, use_container_width=True)


## Write key outputs in the control panel
right_col.subheader("Results and decision")

observed_sessions = data["sessions"].sum()
observed_clicks = data["clicks"].sum()
observed_click_rate = observed_clicks / observed_sessions
worst_case_proba = posterior.cdf(worst_case_threshold)

if worst_case_proba < worst_case_max_proba:
    decision = "GO"
else:
    decision = "NO GO"

right_col.markdown(f"**Observed sessions:** {observed_sessions:,}")
right_col.markdown(f"**Observed click rate:** {observed_click_rate:.4f}")
right_col.markdown(f"**Mean posterior click rate:** {posterior.mean():.4f}")
right_col.markdown(
    f"**80% credible region for click rate:** [{posterior.ppf(0.1):.4f}, {posterior.ppf(0.9):.4f}]"
)
right_col.markdown(
    f"**P(click rate < than critical threshold):** {worst_case_proba:.2%}"
)
right_col.subheader(f"***Final decision: {decision}***")
