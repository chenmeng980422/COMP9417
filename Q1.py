import pandas as pd


data = pd.read_csv('/Users/chenmeng/Downloads/COMP9417/Homework/1/real_estate.csv')
df = pd.DataFrame(data)
df

df[df.isnull().values == True]

df = df.dropna()
df = df.drop(['transactiondate', 'latitude', 'longitude'], axis=1)
df['age_norm'] = (df.age - df.age.min()) / (df.age.max() - df.age.min())
df['nearestMRT_norm'] = (df.nearestMRT - df.nearestMRT.min()) / (df.nearestMRT.max() - df.nearestMRT.min())
df['nConvenience_norm'] = (df.nConvenience - df.nConvenience.min()) / (df.nConvenience.max() - df.nConvenience.min())
df['price_norm'] = (df.price - df.price.min()) / (df.price.max() - df.price.min())
df = df.drop(['age', 'nearestMRT', 'nConvenience', 'price'], axis=1)
df.mean()


