import tkinter as tk
from tkinter import ttk, messagebox
import torch
import pandas as pd
import numpy as np
import joblib
from model import CarPriceNN 

# Učitavanje dataseta
df = pd.read_csv("../data/car_sales_data.csv")

manufacturers = sorted(df["Manufacturer"].unique())
fuel_types = sorted(df["Fuel type"].unique())

# Mapiranje proizvođač -> modeli
models_by_maker = {
    maker: sorted(df.loc[df["Manufacturer"] == maker, "Model"].unique())
    for maker in manufacturers
}

train_ds = torch.load("../data/train_ds.pt", weights_only=False)

# Učitavanje modela
model = CarPriceNN(train_ds.tensors[0].shape[1])
model.load_state_dict(torch.load("../models/car_price_model.pth", weights_only=False))
model.eval()

device = torch.device("cpu")
model.to(device)


# Tkinter GUI
root = tk.Tk()
root.title("Predikcija cene automobila")
root.geometry("400x450")
root.resizable(False, False)

frm = ttk.Frame(root, padding=15)
frm.pack(fill="both", expand=True)

ttk.Label(frm, text="Proizvođač:").grid(column=0, row=0, sticky="w")
manufacturer_cb = ttk.Combobox(frm, values=manufacturers, state="readonly")
manufacturer_cb.grid(column=1, row=0, padx=5, pady=5)

ttk.Label(frm, text="Model:").grid(column=0, row=1, sticky="w")
model_cb = ttk.Combobox(frm, state="disabled")
model_cb.grid(column=1, row=1, padx=5, pady=5)

def on_manufacturer_select(event):
    maker = manufacturer_cb.get()
    model_cb["values"] = models_by_maker[maker]
    model_cb["state"] = "readonly"
    model_cb.set("")

manufacturer_cb.bind("<<ComboboxSelected>>", on_manufacturer_select)

ttk.Label(frm, text="Gorivo:").grid(column=0, row=2, sticky="w")
fuel_cb = ttk.Combobox(frm, values=fuel_types, state="readonly")
fuel_cb.grid(column=1, row=2, padx=5, pady=5)

ttk.Label(frm, text="Godina:").grid(column=0, row=3, sticky="w")
year_entry = ttk.Entry(frm)
year_entry.grid(column=1, row=3, padx=5, pady=5)

ttk.Label(frm, text="Kilometraža (1000km):").grid(column=0, row=4, sticky="w")
mileage_entry = ttk.Entry(frm)
mileage_entry.grid(column=1, row=4, padx=5, pady=5)

ttk.Label(frm, text="Zapremina (L):").grid(column=0, row=5, sticky="w")
engine_entry = ttk.Entry(frm)
engine_entry.grid(column=1, row=5, padx=5, pady=5)

# Učitavanje preprocesora
preprocessor = joblib.load("../models/preprocessor.pkl")


def predict_price():
    try:
        maker = manufacturer_cb.get()
        model_name = model_cb.get()
        fuel = fuel_cb.get()
        year = float(year_entry.get())
        mileage = float(mileage_entry.get()) * 0.6213711922
        engine = float(engine_entry.get())

        if not all([maker, model_name, fuel]):
            messagebox.showerror("Greška", "Popuni sva polja!")
            return

        sample = pd.DataFrame([{
            "Manufacturer": maker,
            "Model": model_name,
            "Fuel type": fuel,
            "Year of manufacture": year,
            "Mileage": mileage,
            "Engine size": engine
        }])

        X_prep = preprocessor.transform(sample)
        X_tensor = torch.tensor(X_prep.toarray() if hasattr(X_prep, "toarray") else X_prep, dtype=torch.float32).to(device)

        with torch.no_grad():
            pred = model(X_tensor).cpu().numpy()[0][0]

        messagebox.showinfo("Predikcija", f"Procenjena cena: {pred:,.2f} €")

    except Exception as e:
        messagebox.showerror("Greška", str(e))


ttk.Button(frm, text="Predvidi cenu", command=predict_price).grid(column=0, row=6, columnspan=2, pady=20)

root.mainloop()

