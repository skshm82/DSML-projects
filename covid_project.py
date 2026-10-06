import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
pd.set_option("display.max_columns", 20)

print("Environment ready.")

cases = pd.read_csv(
    "https://raw.githubusercontent.com/imdevskp/covid-19-india-data/master/complete.csv"
)

vaccination = pd.read_csv(
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/country_data/India.csv"
)

print("Cases shape:", cases.shape)
print("Vaccination shape:", vaccination.shape)

cases.head()

vaccination.shape

cases.dtypes

cases.isna().sum()

print("Unique state/UT names:", cases["Name of State / UT"].nunique())
sorted(cases["Name of State / UT"].unique())

cases.columns = [
    "date",
    "state",
    "lat",
    "long",
    "confirmed",
    "deaths",
    "cured",
    "new_cases",
    "new_deaths",
    "new_recovered"
]

cases["date"] = pd.to_datetime(cases["date"])
cases["deaths"] = pd.to_numeric(
    cases["deaths"], errors="coerce"
).fillna(0).astype(int)
cases["confirmed"] = cases["confirmed"].astype(int)
cases["cured"] = cases["cured"].astype(int)

state_corrections = {
    "Telangana***": "Telangana",
    "Telengana": "Telangana",
    "Union Territory of Jammu and Kashmir": "Jammu and Kashmir",
    "Union Territory of Ladakh": "Ladakh",
    "Union Territory of Chandigarh": "Chandigarh"
}

cases["state"] = cases["state"].replace(state_corrections)

cases = (
    cases
    .drop_duplicates(subset=["date", "state"])
    .sort_values(["state", "date"])
    .reset_index(drop=True)
)

cases["active"] = (
    cases["confirmed"] -
    cases["deaths"] -
    cases["cured"]
)

print("Clean state count:", cases["state"].nunique())
cases.dtypes

cases.isna().sum().sum()

national = (
    cases
    .groupby("date", as_index=False)[["confirmed", "deaths", "cured"]]
    .sum()
)

national["recovery_rate"] = (
    national["cured"] / national["confirmed"] * 100
).round(2)

national["death_rate"] = (
    national["deaths"] / national["confirmed"] * 100
).round(2)

national["new_confirmed"] = (
    national["confirmed"]
    .diff()
    .fillna(national["confirmed"])
)

national.tail()

latest_date = cases["date"].max()
latest_summary = national.iloc[-1]

print(f"As of {latest_date.date()}:")
print(f"Total confirmed : {latest_summary['confirmed']:,}")
print(f"Total recovered : {latest_summary['cured']:,}")
print(f"Total deaths    : {latest_summary['deaths']:,}")
print(f"Recovery rate   : {latest_summary['recovery_rate']}%")
print(f"Death rate      : {latest_summary['death_rate']}%")

latest = cases[cases["date"] == latest_date].copy()

latest["recovery_rate"] = (
    latest["cured"] / latest["confirmed"] * 100
).round(2)

latest["death_rate"] = (
    latest["deaths"] / latest["confirmed"] * 100
).round(2)

latest = (
    latest
    .sort_values("confirmed", ascending=False)
    .reset_index(drop=True)
)

top10 = latest.head(10)

top10[
    [
        "state",
        "confirmed",
        "deaths",
        "cured",
        "recovery_rate",
        "death_rate"
    ]
]

cases["month"] = cases["date"].dt.strftime("%Y-%m")

month_pivot = cases[
    cases["state"].isin(top10["state"].head(5))
].pivot_table(
    values="new_cases",
    index="state",
    columns="month",
    aggfunc="sum",
    fill_value=0
)

month_pivot

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    national["date"],
    national["confirmed"],
    label="Confirmed",
    lw=2
)

ax.plot(
    national["date"],
    national["cured"],
    label="Recovered",
    lw=2
)

ax.plot(
    national["date"],
    national["deaths"],
    label="Deaths",
    lw=2
)

ax.set_title("India: Cumulative COVID-19 Cases")
ax.set_xlabel("Date")
ax.set_ylabel("Number of People")
ax.legend()

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    national["date"],
    national["new_confirmed"],
    width=1.0
)

ax.set_title("India: Daily New Confirmed Cases")
ax.set_xlabel("Date")
ax.set_ylabel("New Cases")

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=top10,
    y="state",
    x="confirmed",
    hue="state",
    palette="Blues_r",
    legend=False,
    ax=ax
)

ax.set_title(
    f"Top 10 States by Confirmed Cases ({latest_date.date()})"
)
ax.set_xlabel("Confirmed Cases")
ax.set_ylabel("")

plt.tight_layout()
plt.show()

top10_sorted = top10.sort_values("recovery_rate")

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=top10_sorted,
    y="state",
    x="recovery_rate",
    hue="state",
    palette="Greens",
    legend=False,
    ax=ax
)

ax.set_title("Recovery Rate of Top 10 Affected States")
ax.set_xlabel("Recovery Rate (%)")
ax.set_ylabel("")

plt.tight_layout()
plt.show()

top5_states = top10["state"].head(5).tolist()

fig, ax = plt.subplots(figsize=(9, 5))

for state_name in top5_states:
    state_data = cases[cases["state"] == state_name]
    ax.plot(
        state_data["date"],
        state_data["confirmed"],
        label=state_name,
        lw=2
    )

ax.set_title("Confirmed Cases Over Time - Top 5 States")
ax.set_xlabel("Date")
ax.set_ylabel("Confirmed Cases")
ax.legend(fontsize=8)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

top6 = latest.nlargest(6, "confirmed")[["state", "confirmed"]]

remaining_cases = (
    latest["confirmed"].sum() -
    top6["confirmed"].sum()
)

pie_labels = top6["state"].tolist() + ["Rest of India"]
pie_values = top6["confirmed"].tolist() + [remaining_cases]

fig, ax = plt.subplots(figsize=(7, 7))

ax.pie(
    pie_values,
    labels=pie_labels,
    autopct="%1.1f%%",
    startangle=90
)

ax.set_title("Share of Total Confirmed Cases")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    national["date"],
    national["recovery_rate"],
    lw=2
)

ax.set_title("India: National Recovery Rate")
ax.set_xlabel("Date")
ax.set_ylabel("Recovery Rate (%)")

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

sizeable = latest[latest["confirmed"] >= 1000]

fig, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(
    data=sizeable,
    x="recovery_rate",
    y="death_rate",
    size="confirmed",
    sizes=(40, 600),
    hue="confirmed",
    palette="Blues",
    legend=False,
    ax=ax
)

for _, row in sizeable.nlargest(6, "confirmed").iterrows():
    ax.annotate(
        row["state"],
        (row["recovery_rate"], row["death_rate"]),
        fontsize=8,
        xytext=(4, 4),
        textcoords="offset points"
    )

ax.set_title("Recovery Rate vs Death Rate by State")
ax.set_xlabel("Recovery Rate (%)")
ax.set_ylabel("Death Rate (%)")

plt.tight_layout()
plt.show()

correlation_columns = [
    "confirmed",
    "deaths",
    "cured",
    "new_confirmed",
    "recovery_rate",
    "death_rate"
]

correlation = national[correlation_columns].corr()

fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=ax
)

ax.set_title("Correlation Between COVID-19 Metrics")

plt.tight_layout()
plt.show()

national["month"] = national["date"].dt.strftime("%Y-%m")

fig, ax = plt.subplots(figsize=(10, 5))

sns.boxplot(
    data=national,
    x="month",
    y="new_confirmed",
    hue="month",
    palette="Blues",
    legend=False,
    ax=ax
)

ax.set_title("Distribution of Daily New Cases by Month")
ax.set_xlabel("Month")
ax.set_ylabel("Daily New Cases")

plt.tight_layout()
plt.show()

vaccination["date"] = pd.to_datetime(vaccination["date"])

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    vaccination["date"],
    vaccination["total_vaccinations"] / 1e9,
    label="Total doses (Bn)",
    lw=2
)

ax.plot(
    vaccination["date"],
    vaccination["people_fully_vaccinated"] / 1e9,
    label="Fully vaccinated (Bn)",
    lw=2
)

ax.set_title("India: COVID-19 Vaccination Progress")
ax.set_xlabel("Date")
ax.set_ylabel("People / Doses (Billions)")
ax.legend()

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

bottom10 = (
    latest[latest["confirmed"] > 0]
    .nsmallest(10, "confirmed")
)

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=bottom10,
    y="state",
    x="confirmed",
    hue="state",
    palette="Oranges",
    legend=False,
    ax=ax
)

ax.set_title(
    f"10 Least-Affected States/UTs ({latest_date.date()})"
)
ax.set_xlabel("Confirmed Cases")
ax.set_ylabel("")

plt.tight_layout()
plt.show()

cases.to_csv("cleaned_covid_india.csv", index=False)

print(
    "Saved cleaned_covid_india.csv:",
    cases.shape[0],
    "rows"
)