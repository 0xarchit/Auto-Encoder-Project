import os
import time
import hashlib
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import gradio as gr
import spaces

# 1. Personal settings
ROLL = "2415800016"
h = int(hashlib.sha256(ROLL.encode()).hexdigest(), 16)
SEED = h % 100000
LATENT = [3, 4, 5, 6][(h >> 4) & 3]

CITIES = [
    ("Chennai", 13.0827, 80.2707), ("Mumbai", 19.0760, 72.8777), ("Kolkata", 22.5726, 88.3639),
    ("Delhi", 28.6139, 77.2090), ("Bengaluru", 12.9716, 77.5946), ("Hyderabad", 17.3850, 78.4867),
    ("Kochi", 9.9312, 76.2673), ("Bhubaneswar", 20.2961, 85.8245), ("Visakhapatnam", 17.6868, 83.2185),
    ("Ahmedabad", 23.0225, 72.5714), ("Jaipur", 26.9124, 75.7873), ("Guwahati", 26.1445, 91.7362),
    ("Lucknow", 26.8467, 80.9462), ("Pune", 18.5204, 73.8567), ("Madurai", 9.9252, 78.1198),
    ("Coimbatore", 11.0168, 76.9558),
]
CITY, LAT, LON = CITIES[(h >> 24) % len(CITIES)]
WINDOW = [5, 7, 10][(h >> 30) % 3]

# 2. Denoising Autoencoder (Tab 1)
class AE(nn.Module):
    def __init__(self, z=LATENT):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, z),
        )
        self.dec = nn.Sequential(
            nn.Linear(z, 64), nn.ReLU(), nn.Linear(64, 256),
            nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z).view(-1, 1, 28, 28), z

test_ds = torchvision.datasets.MNIST("data", train=False, download=True)
Xte = (test_ds.data.float() / 255.0).unsqueeze(1)

dae_gpu = None

def get_dae_gpu():
    global dae_gpu
    if dae_gpu is not None:
        return dae_gpu

    model = AE().to("cuda")
    checkpoint = f"dae_{ROLL}.pt"
    if os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location="cuda")
        model.load_state_dict(state)
    else:
        torch.manual_seed(SEED)
        train_ds = torchvision.datasets.MNIST("data", train=True, download=True)
        x_train = (train_ds.data.float() / 255.0).unsqueeze(1).to("cuda")
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        model.train()
        for _ in range(3):
            for start in range(0, len(x_train), 256):
                x = x_train[start:start + 256]
                noisy = (x + 0.5 * torch.randn_like(x)).clamp(0, 1)
                loss = F.mse_loss(model(noisy)[0], x)
                opt.zero_grad()
                loss.backward()
                opt.step()
        torch.save(model.state_dict(), checkpoint)

    model.eval()
    dae_gpu = model
    return dae_gpu

def to28(img):
    arr = np.asarray(
        Image.fromarray(np.asarray(img).astype(np.uint8)).convert("L").resize((28, 28)),
        dtype=np.float32,
    ) / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    return torch.from_numpy(arr.copy())[None, None]

def big_from_gpu(t_gpu):
    arr = t_gpu[0, 0].detach().clamp(0, 1).cpu().numpy()
    return (arr * 255).astype(np.uint8).repeat(8, 0).repeat(8, 1)

@spaces.GPU
def denoise(img, idx, sigma):
    model = get_dae_gpu()
    x_cpu = to28(img) if img is not None else Xte[int(idx)][None]
    x_gpu = x_cpu.to("cuda")
    gen = torch.Generator(device="cuda").manual_seed(0)
    noisy_gpu = (x_gpu + float(sigma) * torch.randn(x_gpu.shape, device="cuda", generator=gen)).clamp(0, 1)
    with torch.inference_mode():
        recon_gpu, _ = model(noisy_gpu)
    return big_from_gpu(x_gpu), big_from_gpu(noisy_gpu), big_from_gpu(recon_gpu)

# 3. Weather Anomaly Model (Tab 2)
VARS = [
    "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
]
ATTRIB = "Weather data by Open-Meteo.com (CC BY 4.0)"

def open_meteo(url, **params):
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    payload = r.json()
    frame = pd.DataFrame(payload["daily"])
    frame["time"] = pd.to_datetime(frame["time"])
    return frame.set_index("time")[VARS].ffill().bfill(), payload.get("daily_units", {})

def transform(frame):
    res = frame[VARS].copy()
    res["precipitation_sum"] = np.log1p(res["precipitation_sum"].clip(lower=0))
    return res

history, _ = open_meteo(
    "https://archive-api.open-meteo.com/v1/archive",
    latitude=LAT, longitude=LON,
    start_date="1991-01-01", end_date="2024-12-31",
    daily=",".join(VARS), timezone="Asia/Kolkata",
)
T = transform(history)
tr_mask = T.index.year <= 2014
monthly_mean = T[tr_mask].groupby(T[tr_mask].index.month).mean()
monthly_std = T[tr_mask].groupby(T[tr_mask].index.month).std() + 1e-6

def standardize(frame):
    months = frame.index.month
    return (frame - monthly_mean.loc[months].values) / monthly_std.loc[months].values

def make_windows(frame, width=WINDOW):
    values = frame.values.astype(np.float32)
    count = len(values) - width + 1
    wins = np.stack([values[i:i + width].ravel() for i in range(count)])
    return wins, frame.index[:count]

Xw, starts = make_windows(standardize(T))
D = Xw.shape[1]

wae_gpu = None
thresh_val = None

def get_wae_and_thresh():
    global wae_gpu, thresh_val
    if wae_gpu is not None and thresh_val is not None:
        return wae_gpu, thresh_val

    Xw_t = torch.from_numpy(Xw.copy()).to("cuda")
    tr_idx = torch.from_numpy(np.asarray(starts.year <= 2014, dtype=np.bool_)).to("cuda")
    train_w = Xw_t[tr_idx]

    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, 6),
        nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, D),
    ).to("cuda")
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    model.train()
    for _ in range(40):
        perm = torch.randperm(len(train_w), device="cuda")
        for start in range(0, len(perm), 256):
            batch = train_w[perm[start:start + 256]]
            loss = F.mse_loss(model(batch), batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.inference_mode():
        err = ((model(Xw_t) - Xw_t) ** 2).view(len(Xw), WINDOW, len(VARS)).cpu().numpy()
    scores = err.mean(axis=(1, 2))
    thresh_val = float(np.quantile(scores[starts.year <= 2014], 0.99))
    wae_gpu = model
    return wae_gpu, thresh_val

def live_window():
    live, _ = open_meteo(
        "https://api.open-meteo.com/v1/forecast",
        latitude=LAT, longitude=LON,
        daily=",".join(VARS), timezone="Asia/Kolkata",
        past_days=WINDOW, forecast_days=1,
    )
    return live.tail(WINDOW)

@spaces.GPU
def weather():
    model, thresh = get_wae_and_thresh()
    live = live_window()
    values = standardize(transform(live)).values.astype(np.float32).ravel()
    x_gpu = torch.from_numpy(values.copy()[None]).to("cuda")
    with torch.inference_mode():
        sc = F.mse_loss(model(x_gpu), x_gpu).item()

    fig, ax = plt.subplots(figsize=(7, 3))
    labels = live.index.strftime("%d %b")
    ax.bar(labels, live["precipitation_sum"], label="Rain (mm)")
    ax.set_ylabel("Rain (mm)")
    ax.set_xlabel("Date")

    ax2 = ax.twinx()
    ax2.plot(labels, live["temperature_2m_max"], "r-o", label="Tmax")
    ax2.set_ylabel("Tmax (°C)")
    ax.set_title(ATTRIB, fontsize=8)
    fig.tight_layout()

    status = "⚠️ **UNUSUAL**" if sc > thresh else "✅ **Normal**"
    verdict = f"**{CITY}**, last {WINDOW} days: score **{sc:.3f}** vs threshold **{thresh:.3f}** → {status}"
    return verdict, fig

# 4. Feedback Logger (Tab 3)
FB = f"feedback_{ROLL}.csv"

def feedback(roll, stars, comment):
    visitor_roll = str(roll or "").strip().upper()
    if not visitor_roll:
        raise gr.Error("Please enter your roll number.")

    row = pd.DataFrame([{
        "time": time.strftime("%Y-%m-%d %H:%M"),
        "visitor_roll": visitor_roll,
        "stars": int(stars),
        "comment": str(comment or "").strip(),
    }])

    if os.path.exists(FB):
        try:
            existing = pd.read_csv(FB)
            result = pd.concat([existing, row], ignore_index=True)
        except Exception:
            result = row
    else:
        result = row

    result.to_csv(FB, index=False)
    return result

# 5. Gradio Interface
with gr.Blocks(title=f"{ROLL} · Autoencoder app") as demo:
    gr.Markdown(f"# 🧠 {ROLL}'s Autoencoder App\nBuilt in LearnLab Module 3")

    with gr.Tab("Denoiser"):
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload a digit photo (optional)")
            with gr.Column():
                image_index = gr.Slider(0, len(Xte) - 1, value=7, step=1, label="Or pick MNIST test image #")
                noise_sigma = gr.Slider(0, 1, value=0.5, step=0.05, label="Noise σ")
                denoise_button = gr.Button("Denoise", variant="primary")
        with gr.Row():
            clean_output = gr.Image(label="Clean")
            noisy_output = gr.Image(label="Noisy")
            denoised_output = gr.Image(label="Denoised")
        denoise_button.click(
            fn=denoise,
            inputs=[image_input, image_index, noise_sigma],
            outputs=[clean_output, noisy_output, denoised_output],
        )

    with gr.Tab(f"Weather alarm · {CITY}"):
        gr.Markdown(f"Anomaly detection for **{CITY}** using the previous {WINDOW} days.")
        weather_button = gr.Button("Check the last days (live Open-Meteo)", variant="primary")
        weather_verdict = gr.Markdown()
        weather_plot = gr.Plot()
        weather_button.click(
            fn=weather,
            inputs=None,
            outputs=[weather_verdict, weather_plot],
        )

    with gr.Tab("Feedback"):
        roll_input = gr.Textbox(label="Your roll number")
        stars_input = gr.Slider(1, 5, value=4, step=1, label="Stars")
        comment_input = gr.Textbox(label="Comment")
        feedback_button = gr.Button("Submit", variant="primary")
        feedback_table = gr.Dataframe()
        feedback_button.click(
            fn=feedback,
            inputs=[roll_input, stars_input, comment_input],
            outputs=feedback_table,
        )

if __name__ == "__main__":
    demo.queue().launch()