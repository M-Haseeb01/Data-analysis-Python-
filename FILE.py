import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate Data
np.random.seed(0)


data = {
    'RAM': np.random.randint(4, 17, size=10),  # RAM between 4GB and 16GB
    'Processor Speed': np.random.randint(1, 4, size=10),  # Processor Speed between 1GHz and 3GHz
    'Battery Life': np.random.randint(2, 11, size=10)  # Battery Life between 2 hours and 10 hours
}


df = pd.DataFrame(data)

df['Price'] = 100 * df['RAM'] + 500 * df['Processor Speed'] + 200 * df['Battery Life'] + np.random.randn(10) * 100

#  Train a Machine Learning Model

X = df[['RAM', 'Processor Speed', 'Battery Life']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Output the results
print(f'Mean Squared Error on Test Set: {mse}')
print(f'Predicted vs Actual Prices:\n{pd.DataFrame({"Predicted": y_pred, "Actual": y_test})}')

plt.figure(figsize=(8, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
sns.scatterplot(x='RAM', y='Price', data=df, ax=axs[0])
sns.scatterplot(x='Processor Speed', y='Price', data=df, ax=axs[1])
sns.scatterplot(x='Battery Life', y='Price', data=df, ax=axs[2])
axs[0].set_title('RAM vs Price')
axs[1].set_title('Processor Speed vs Price')
axs[2].set_title('Battery Life vs Price')
plt.show()



