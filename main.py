# --- Data Download & Feature Engineering ---

# Download historical stock data for given assets and date range
def fetch_data(assets, start, end=None):
    df = yf.download(assets, start=start, end=end)
    # Defensive: check for 'Adj Close' in columns
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.get_level_values(0):
            df = df['Adj Close']
        elif 'Close' in df.columns.get_level_values(0):
            df = df['Close']
        else:
            raise KeyError(f"Neither 'Adj Close' nor 'Close' found in downloaded data. Columns: {df.columns}")
    else:
        if 'Adj Close' in df.columns:
            df = df[['Adj Close']]
            df.columns = [assets[0]]
        elif 'Close' in df.columns:
            df = df[['Close']]
            df.columns = [assets[0]]
        else:
            raise KeyError(f"Neither 'Adj Close' nor 'Close' found in downloaded data. Columns: {df.columns}")
    return df.dropna()

# Map install name to import name if different
IMPORT_NAMES = {'scikit-learn': 'sklearn'}
REQUIRED = [
    'numpy', 'pandas', 'matplotlib', 'plotly', 'torch', 'scikit-learn', 'yfinance'
]

import sys
import subprocess
import importlib
import os

# --- Dependency Auto-Installer ---
REQUIRED = [
    'numpy', 'pandas', 'matplotlib', 'plotly', 'torch', 'scikit-learn', 'yfinance'
]

def install_and_import(package):

    import_name = IMPORT_NAMES.get(package, package)
    try:
        module = importlib.import_module(import_name)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        module = importlib.import_module(import_name)
    globals()[import_name] = module

for pkg in REQUIRED:
    install_and_import(pkg)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import yfinance as yf

# --- Config ---
ASSETS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'UNH']
START_DATE = '2018-01-01'
END_DATE = None  # today
ROLL_VOL = 20
MA_WINDOWS = [10, 20, 50]
LATENT_DIM = 2
BATCH_SIZE = 64
EPOCHS = 60
SEED = 42  # For reproducibility
# --- Data Download & Feature Engineering ---


# Map install name to import name if different
IMPORT_NAMES = {'scikit-learn': 'sklearn'}
REQUIRED = [
    'numpy', 'pandas', 'matplotlib', 'plotly', 'torch', 'scikit-learn', 'yfinance'
]


def compute_features(df):
    feats = []
    for asset in df.columns:
        px = df[asset]
        ret = px.pct_change().fillna(0)
        vol = ret.rolling(ROLL_VOL).std().fillna(0)
        mas = [px.rolling(w).mean().fillna(0) for w in MA_WINDOWS]
        asset_df = pd.DataFrame({
            'return': ret,
            'vol': vol,
            **{f'ma{w}': ma for w, ma in zip(MA_WINDOWS, mas)}
        })
        asset_df['asset'] = asset
        feats.append(asset_df)
    feats = pd.concat(feats)
    feats['time'] = np.tile(df.index, len(df.columns))
    feats = feats.reset_index(drop=True)
    return feats

def normalize_features(feats):
    scaler = StandardScaler()
    X = scaler.fit_transform(feats[['return', 'vol'] + [f'ma{w}' for w in MA_WINDOWS]])
    return X, scaler

# --- VAE Model ---
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld

# --- Training ---
def train_vae(X, latent_dim=2, epochs=60, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vae = VAE(X.shape[1], latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, mu, logvar = vae(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(X):.4f}")
    return vae

# --- Latent Space Projection ---
def get_latent(vae, X):
    device = next(vae.parameters()).device
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        mu, _ = vae.encode(X_tensor)
        return mu.cpu().numpy()

# --- Clustering for Regimes ---
def cluster_regimes(latent, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    return kmeans.fit_predict(latent)

# --- Visualization ---
def animate_latent(feats, latent, regimes, color_by='vol', title='VAE Latent Space'):
    feats = feats.copy()
    feats['z1'] = latent[:,0]
    feats['z2'] = latent[:,1]
    feats['regime'] = regimes
    # Animate by time
    times = sorted(feats['time'].unique())
    frames = []
    for t in times:
        frame = feats[feats['time']==t]
        frames.append(frame)
    fig = px.scatter(
        frames[0], x='z1', y='z2', color=color_by, symbol='asset',
        color_continuous_scale=px.colors.sequential.Plasma,
        title=title, template='plotly_dark',
        hover_data=['asset', 'vol', 'regime', 'time']
    )
    # Trajectory lines
    for asset in feats['asset'].unique():
        asset_traj = feats[feats['asset']==asset].sort_values('time')
        fig.add_trace(go.Scatter(
            x=asset_traj['z1'], y=asset_traj['z2'],
            mode='lines', line=dict(width=1),
            name=f'{asset} path',
            opacity=0.3, showlegend=False
        ))
    # Animation frames
    fig.frames = [go.Frame(
        data=[go.Scatter(
            x=frame['z1'], y=frame['z2'],
            mode='markers',
            marker=dict(
                color=frame[color_by],
                colorscale='Plasma',
                size=12,
                line=dict(width=1, color='white')
            ),
            text=frame['asset'],
            customdata=np.stack([frame['asset'], frame['vol'], frame['regime'], frame['time']], axis=-1),
            hovertemplate='<b>%{customdata[0]}</b><br>Vol: %{customdata[1]:.3f}<br>Regime: %{customdata[2]}<br>Time: %{customdata[3]}'
        )], name=str(t)
    ) for t, frame in zip(times, frames)]
    fig.update_layout(
        updatemenus=[dict(
            type='buttons', showactive=False,
            buttons=[dict(label='Play', method='animate', args=[None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]),
                     dict(label='Pause', method='animate', args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}])]
        )]
    )
    fig.show()

# --- Synthetic Scenario Generation ---
def sample_and_generate(vae, scaler, n=100):
    device = next(vae.parameters()).device
    z = torch.randn(n, LATENT_DIM).to(device)
    with torch.no_grad():
        gen = vae.decode(z).cpu().numpy()
    # Inverse transform to feature space
    gen_feats = scaler.inverse_transform(gen)
    return gen_feats

def animate_generated_vs_real(real_feats, gen_feats):
    # Only plot returns vs vol for simplicity
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=real_feats[:,0], y=real_feats[:,1],
        mode='markers', name='Real',
        marker=dict(color='deepskyblue', size=8, opacity=0.5)
    ))
    fig.add_trace(go.Scatter(
        x=gen_feats[:,0], y=gen_feats[:,1],
        mode='markers', name='Generated',
        marker=dict(color='orange', size=8, opacity=0.5)
    ))
    fig.update_layout(
        title='Generated vs Real Market Scenarios',
        xaxis_title='Return', yaxis_title='Volatility',
        template='plotly_dark'
    )
    fig.show()

# --- Main Pipeline ---
def main():
    print('Fetching data...')
    df = fetch_data(ASSETS, START_DATE, END_DATE)
    print('Computing features...')
    feats = compute_features(df)
    print('Normalizing...')
    X, scaler = normalize_features(feats)
    print('Training VAE...')
    vae = train_vae(X, latent_dim=LATENT_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE)
    print('Projecting to latent space...')
    latent = get_latent(vae, X)
    print('Clustering regimes...')
    regimes = cluster_regimes(latent, n_clusters=4)
    print('Animating latent space...')
    animate_latent(feats, latent, regimes, color_by='vol', title='VAE Latent Space: Volatility')
    print('Animating by regime...')
    animate_latent(feats, latent, regimes, color_by='regime', title='VAE Latent Space: Regime')
    print('Generating synthetic scenarios...')
    gen_feats = sample_and_generate(vae, scaler, n=100)
    print('Animating generated vs real...')
    animate_generated_vs_real(X, gen_feats)

if __name__ == '__main__':
    main()
