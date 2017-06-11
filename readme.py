import guacml as guac

# wrapper around Pandas + memory about ML models
ds = guac.GuacMl('data/training_set.csv')

ds.run_autopilot #?
ds.preprocess (it describes at the end)
ds.describe
ds.benchmark

ds.impute(features=None, method=...)
ds.room_count.impute(method)

ds.room_count.describe
