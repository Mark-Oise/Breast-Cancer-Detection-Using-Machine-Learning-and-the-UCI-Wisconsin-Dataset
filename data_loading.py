from ucimlrepo import fetch_ucirepo

# Fetch dataset
dataset = fetch_ucirepo(id=17)

# data (as pandas dataframes) 
X = dataset.data.features
y = dataset.data.targets

# metadata
print(dataset.metadata)

# variable information
print(dataset.variables)




