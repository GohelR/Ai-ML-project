import pandas as pd
import zipfile

# Path to the ZIP file
zip_path = r'D:\whatsapp files\archive.zip'

# Extract the specific CSV file from the ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extract('nsl-kdd/KDDTrain+.txt', 'D:/extracted_files')

# Load the extracted CSV file
csv_path = r'D:/extracted_files/nsl-kdd/KDDTrain+.txt'
data = pd.read_csv(csv_path)
print(data.head())
print(data.head())
print(data.info())
# Example: Fill missing values with the mean for numerical columns
data.fillna(data.mean(), inplace=True)
# Example: Select a subset of features
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
X = data[features]
y = data['SalePrice']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

