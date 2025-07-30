import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DummyModel:
    def predict(self, X):
        # Return random values similar to ATM cash amount predictions
        return np.clip(np.random.normal(70000, 15000, size=len(X)), 20000, 120000)
model = joblib.load("atm_model.pkl")


states = [
    "Maharashtra", "Uttar Pradesh", "Karnataka", "Tamil Nadu", "Delhi",
    "Gujarat", "West Bengal", "Madhya Pradesh", "Rajasthan", "Bihar"
]

location_map = {"Urban": 0, "Suburban": 1, "Rural": 2, "All": -1}

state_data = {}

# Generate ATM data
def generate_atm_data(n, day, holiday, loc_type):
    data = []
    for _ in range(n):
        avg = np.clip(np.random.normal(70000, 15000), 20000, 120000)
        loc = location_map[loc_type] if loc_type != "All" else np.random.choice([0, 1, 2])
        data.append([day, holiday, avg, loc])
    return np.array(data)

# Predict cash
def predict():
    result_table.delete(*result_table.get_children())
    state_data.clear()

    selected = [states_listbox.get(i) for i in states_listbox.curselection()]
    if not selected:
        messagebox.showwarning("No State Selected", "Please select one or more states.")
        return

    try:
        n = int(atm_count_var.get())
        if n <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a positive integer for ATMs per state.")
        return

    day = day_var.get()
    holiday = int(holiday_var.get())
    loc_type = location_type_var.get()

    for state in selected:
        features = generate_atm_data(n, day, holiday, loc_type)
        predictions = model.predict(features)
        total = int(predictions.sum())
        avg = int(predictions.mean())
        std = int(np.std(predictions))
        max_ = int(predictions.max())
        min_ = int(predictions.min())
        loc_counts = {
            "Urban": sum(1 for row in features if row[3] == 0),
            "Suburban": sum(1 for row in features if row[3] == 1),
            "Rural": sum(1 for row in features if row[3] == 2)
        }
        state_data[state] = {
            "total": total, "avg": avg, "std": std, "max": max_, "min": min_,
            "loc_dist": loc_counts, "predictions": predictions
        }
        result_table.insert("", "end", values=(state, f"₹{total:,}", f"₹{avg:,}", f"₹{max_:,}", f"₹{min_:,}", f"₹{std:,}"))

    update_all_charts()
    update_summary()

def update_all_charts():
    update_total_chart()
    update_pie_chart()
    update_histogram()

def update_total_chart():
    chart_ax1.clear()
    names = list(state_data.keys())
    values = [d["total"] for d in state_data.values()]
    if chart_type.get() == "Bar":
        chart_ax1.bar(names, values, color="#1f77b4")
    else:
        chart_ax1.plot(names, values, marker='o', color="#2ca02c")
    chart_ax1.set_title("ATM Total Cash by State")
    chart_ax1.set_ylabel("INR")
    chart_ax1.set_xticklabels(names, rotation=45, ha='right')
    canvas1.draw()

def update_pie_chart():
    chart_ax2.clear()
    loc_counter = {"Urban": 0, "Suburban": 0, "Rural": 0}
    for state in state_data.values():
        for loc, count in state["loc_dist"].items():
            loc_counter[loc] += count
    labels, sizes = zip(*loc_counter.items())
    chart_ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
    chart_ax2.set_title("ATM Location Distribution")
    canvas2.draw()

def update_histogram():
    chart_ax3.clear()
    if not state_data:
        chart_ax3.text(0.5, 0.5, "No data to show", ha='center', va='center', fontsize=14)
        canvas3.draw()
        return
    all_preds = np.concatenate([d["predictions"] for d in state_data.values()])
    chart_ax3.hist(all_preds, bins=15, color="#d62728", alpha=0.8)
    chart_ax3.set_title("Prediction Distribution")
    chart_ax3.set_xlabel("Cash to Load (INR)")
    canvas3.draw()

def update_summary():
    if not state_data:
        summary_label.config(text="")
        return
    totals = [d["total"] for d in state_data.values()]
    summary_label.config(text=f" Total Cash: ₹{sum(totals):,}   |   Max: ₹{max(totals):,}   |   Min: ₹{min(totals):,}")

def export_to_csv():
    if not state_data:
        messagebox.showinfo("No Data", "No data to export. Please run prediction first.")
        return
    df = pd.DataFrame([
        {
            "State": s,
            "Total Cash": d["total"],
            "Avg/ATM": d["avg"],
            "Max": d["max"],
            "Min": d["min"],
            "Std Dev": d["std"],
            "Urban ATMs": d["loc_dist"]["Urban"],
            "Suburban ATMs": d["loc_dist"]["Suburban"],
            "Rural ATMs": d["loc_dist"]["Rural"]
        } for s, d in state_data.items()
    ])
    file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file:
        df.to_csv(file, index=False)
        messagebox.showinfo("Export Successful", f"Data exported to:\n{file}")

# --- Hover effect for buttons ---
def on_enter(e):
    e.widget['background'] = '#005f99'
def on_leave(e):
    e.widget['background'] = '#007acc'

# Scrollable Frame Class
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        self.canvas = tk.Canvas(self, bg="#e8f0fe", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mousewheel scrolling support
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

# --- Main Window ---
root = tk.Tk()
root.title("🇮🇳 Modern ATM Cash Dashboard with Scroll")
root.geometry("1200x900")
root.configure(bg="#e8f0fe")

scroll_frame = ScrollableFrame(root)
scroll_frame.pack(fill="both", expand=True)

content = scroll_frame.scrollable_frame

# Fonts & Styles
title_font = ("Segoe UI", 20, "bold")
label_font = ("Segoe UI", 11)
btn_font = ("Segoe UI", 12, "bold")

# Title
title = tk.Label(content, text=" Indian State-wise ATM Cash Forecast", font=title_font, bg="#e8f0fe", fg="#0a3d62")
title.pack(pady=15)

# --- Frame for controls ---
control_frame = tk.Frame(content, bg="#e8f0fe")
control_frame.pack(pady=10, padx=10, fill='x')

# States selection with scrollbar
states_frame = tk.Frame(control_frame, bg="#e8f0fe")
states_frame.grid(row=0, column=0, rowspan=6, sticky="ns")

tk.Label(states_frame, text="Select States:", bg="#e8f0fe", font=label_font).pack(anchor='w')

# Listbox + Scrollbar
scrollbar_states = tk.Scrollbar(states_frame, orient=tk.VERTICAL)
states_listbox = tk.Listbox(states_frame, selectmode="multiple", height=8, yscrollcommand=scrollbar_states.set, font=label_font, width=20)
for s in states:
    states_listbox.insert(tk.END, s)
scrollbar_states.config(command=states_listbox.yview)
states_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar_states.pack(side=tk.RIGHT, fill=tk.Y)

# Other controls
tk.Label(control_frame, text="ATMs per State:", bg="#e8f0fe", font=label_font).grid(row=0, column=1, padx=20, sticky="w")
atm_count_var = tk.StringVar(value="10")
tk.Entry(control_frame, textvariable=atm_count_var, width=8, font=label_font).grid(row=0, column=2, sticky="w")

tk.Label(control_frame, text="Day of Week:", bg="#e8f0fe", font=label_font).grid(row=1, column=1, padx=20, sticky="w")
day_var = tk.IntVar(value=0)
tk.OptionMenu(control_frame, day_var, *list(range(7))).grid(row=1, column=2, sticky="w")

tk.Label(control_frame, text="Holiday:", bg="#e8f0fe", font=label_font).grid(row=2, column=1, padx=20, sticky="w")
holiday_var = tk.StringVar(value="0")
tk.OptionMenu(control_frame, holiday_var, "0", "1").grid(row=2, column=2, sticky="w")

tk.Label(control_frame, text="Location Type:", bg="#e8f0fe", font=label_font).grid(row=3, column=1, padx=20, sticky="w")
location_type_var = tk.StringVar(value="All")
tk.OptionMenu(control_frame, location_type_var, *location_map.keys()).grid(row=3, column=2, sticky="w")

tk.Label(control_frame, text="Chart Type:", bg="#e8f0fe", font=label_font).grid(row=4, column=1, padx=20, sticky="w")
chart_type = tk.StringVar(value="Bar")
tk.OptionMenu(control_frame, chart_type, "Bar", "Line").grid(row=4, column=2, sticky="w")

# Buttons
btn_predict = tk.Button(content, text=" Predict", bg="#007acc", fg="white", font=btn_font, width=15, command=predict, borderwidth=0, relief="raised")
btn_predict.pack(pady=10)
btn_predict.bind("<Enter>", on_enter)
btn_predict.bind("<Leave>", on_leave)

btn_export = tk.Button(content, text="Export to CSV", bg="#00965f", fg="white", font=btn_font, width=15, command=export_to_csv, borderwidth=0, relief="raised")
btn_export.pack()
btn_export.bind("<Enter>", lambda e: e.widget.config(bg="#007a4d"))
btn_export.bind("<Leave>", lambda e: e.widget.config(bg="#00965f"))

# Results table
columns = ("State", "Total", "Avg/ATM", "Max", "Min", "Std Dev")
result_table = ttk.Treeview(content, columns=columns, show="headings", height=8)
for col in columns:
    result_table.heading(col, text=col)
    result_table.column(col, width=140, anchor="center")
result_table.pack(pady=20)

# Summary Label
summary_label = tk.Label(content, text="", font=("Segoe UI", 13, "bold"), bg="#e8f0fe", fg="#0a3d62")
summary_label.pack()

# Charts Frame
chart_frame = tk.Frame(content, bg="#e8f0fe")
chart_frame.pack(pady=10)

fig1, chart_ax1 = plt.subplots(figsize=(5, 3))
canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame)
canvas1.get_tk_widget().grid(row=0, column=0, padx=15)

fig2, chart_ax2 = plt.subplots(figsize=(4, 3))
canvas2 = FigureCanvasTkAgg(fig2, master=chart_frame)
canvas2.get_tk_widget().grid(row=0, column=1, padx=15)

fig3, chart_ax3 = plt.subplots(figsize=(5, 3))
canvas3 = FigureCanvasTkAgg(fig3, master=chart_frame)
canvas3.get_tk_widget().grid(row=1, column=0, columnspan=2, pady=15)

root.mainloop()
