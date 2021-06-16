
>  # ⚠️⛔️⚠️ Currently Unmaintained ⚠️⛔️⚠️
>
> We lost interest in the topic a while ago and are no longer maintainting guacml. Feel free to reach out if you still want to contribute or take over.


# Guacamole - ML without the boilerplate

Automate typical DS/ML workflow, get rid of the boilerplate, use sensible defaults, but expose ways to customize them ("convention over configuration").

We start with a focus on tabular data like typical business spreadsheets.

1. Load data
  - CSV
  - ... (later versions)
2. Explore
  - Automated plots and table of stats for each feature
3. Clean
4. Engineer features
5. Train and compare models
6. Repeat
7. Deploy

Ideally, steps 2-3 should be semi-automated (part of 4 as well).

## How to get started

1. Clone the project to your own machine
2. Run `scripts/setup` - This will install xgboost and all python dependencies
3. Run `scripts/demo` - This will run a jupyter notebook server so you can take a look at the demo notebooks
