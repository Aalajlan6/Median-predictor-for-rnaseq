import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
data = pd.read_csv('AI training.csv')
X = data.drop(columns=['GENE ID', 'LOCUS TAG', 'PRODUCT NAME', 'SCAFFOLD ID', 'READS COUNT', 'GENE SEQ LENGTH', 'COVERAGE', 'READS COUNT ANTISENSE', 'COVERAGE ANTISENSE', 'MEDIAN ANTISENSE'])
y = data['MEDIAN']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)


loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')


model.save('medianPrediction.keras')
"""

model = load_model('medianPrediction.keras')
predictions = model.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()
r2 = r2_score(y_test, predictions)
print(f'R² Score: {r2}')

new_data = pd.read_csv('Test2.csv')
X_new = new_data.drop(columns=['GENE ID', 'LOCUS TAG', 'PRODUCT NAME', 'SCAFFOLD ID', 'READS COUNT', 'GENE SEQ LENGTH', 'COVERAGE', 'READS COUNT ANTISENSE', 'COVERAGE ANTISENSE', 'MEDIAN ANTISENSE'])
new_predictions = model.predict(X_new)

new_data['Predicted MEDIAN'] = new_predictions
new_data.to_csv('test2_with_predictions.csv', index=False)

print(new_predictions)

