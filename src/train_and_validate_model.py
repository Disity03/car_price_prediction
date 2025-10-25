import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CarPriceNN
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import datetime

# Učitavanje podataka
train_ds = torch.load("../data/train_ds.pt", weights_only=False)
val_ds = torch.load("../data/val_ds.pt", weights_only=False)
test_ds = torch.load("../data/test_ds.pt", weights_only=False)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=256)
test_dl = DataLoader(test_ds, batch_size=256)

# Trening modela
device = torch.device("cpu")

model = CarPriceNN(train_ds.tensors[0].shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []
n_epochs = 200
filename = f"../outputs/training_report_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
f = open(filename,"a")
for epoch in range(n_epochs):
	model.train()
	train_loss = 0
	
	for xb, yb in train_dl:
		xb, yb = xb.to(device), yb.to(device)
		optimizer.zero_grad()
		preds = model(xb)
		loss = criterion(preds, yb)
		loss.backward()
		optimizer.step()
		train_loss += loss.item() * xb.size(0)
	
	avg_train_loss = train_loss / len(train_dl.dataset)
	
	model.eval()
	val_loss = 0
	
	with torch.no_grad():
		for xb, yb in val_dl:
			xb, yb = xb.to(device), yb.to(device)
			preds = model(xb)
			loss = criterion(preds, yb)
			val_loss += loss.item() * xb.size(0)
	
	avg_val_loss = val_loss / len(val_dl.dataset)
	print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")
	f.write(f"Epoch {epoch+1:2d}/{n_epochs} - Train Loss: {avg_train_loss:9.4f} - Validation Loss: {avg_val_loss:9.4f}\n")
	if epoch > 10: 
		train_losses.append(avg_train_loss)
		val_losses.append(avg_val_loss)

# Čuvanje modela
torch.save(model.state_dict(), "../models/car_price_model.pth")

# Rezultati testa
model.eval()
with torch.no_grad():
	preds = model(test_ds.tensors[0].to(device)).cpu().numpy()

mae = mean_absolute_error(test_ds.tensors[1].numpy(), preds)
mse = mean_squared_error(test_ds.tensors[1].numpy(), preds)
r2 = r2_score(test_ds.tensors[1].numpy(), preds)

rmse = np.sqrt(mse)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

# Graficki prikaz za Train vs Validation Loss
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.show()
