import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import TensorDataset
import joblib

df = pd.read_csv("../data/car_sales_data.csv")

X = df.drop(columns=["Price"])
y = df["Price"]

# Odredi tipove kolona
categorical = X.select_dtypes(include=["object"]).columns
numerical = X.select_dtypes(exclude=["object"]).columns

# Preprocesiranje 
preprocessor = ColumnTransformer([
	("num", StandardScaler(), numerical),
	("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

X_prep = preprocessor.fit_transform(X)

# Čuvanje preprocesora
joblib.dump(preprocessor, "../models/preprocessor.pkl")

# Pretvori u torch tenzore
X_train, X_tmp, y_train, y_tmp = train_test_split(X_prep, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

X_train = torch.tensor(X_train.toarray() if hasattr(X_train, "toarray") else X_train, dtype=torch.float32)
X_val = torch.tensor(X_val.toarray() if hasattr(X_val, "toarray") else X_val, dtype=torch.float32)
X_test  = torch.tensor(X_test.toarray() if hasattr(X_test, "toarray") else X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
y_test  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# DataLoader
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

# Čuvanje u fajlove
torch.save(train_ds, "../data/train_ds.pt")
torch.save(val_ds, "../data/val_ds.pt")
torch.save(test_ds, "../data/test_ds.pt")

print("Data is ready!")
