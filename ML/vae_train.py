import os
import gc
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from pathlib import Path
from dataclasses import dataclass
from collections import deque
import multiprocessing as mp

ffp_arr = np.load('ffp_arr_64.npy')
auto_arr = np.load('auto_arr_64.npy')

# # DeepLearning Part

# #### Model 


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len=1024, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len):
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x, input_pos=None):
        seq_len = x.size(1)
        if input_pos is None:
            input_pos = torch.arange(seq_len, device=x.device)
        input_pos = input_pos.to(device=self.cache.device, dtype=torch.long)
        cache = self.cache[input_pos]
        if cache.dim() == 3:
            cache = cache.unsqueeze(0)
        cache = cache.unsqueeze(2)
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x_out = torch.stack([
            xshaped[..., 0] * cache[..., 0] - xshaped[..., 1] * cache[..., 1],
            xshaped[..., 0] * cache[..., 1] + xshaped[..., 1] * cache[..., 0],
        ], dim=-1)
        return x_out.flatten(-2).type_as(x)


class RoPEAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, max_seq_len=1024, base=10000):
        super().__init__()
        assert d_model % nhead == 0
        assert (d_model // nhead) % 2 == 0, "RoPE requires even head_dim"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=max_seq_len, base=base)

    def forward(self, x, mask=None, input_pos=None):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.nhead, self.head_dim).unbind(dim=2)
        q = self.rope(q, input_pos)
        k = self.rope(k, input_pos)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask[:, None, :, :]
            if mask.dtype == torch.bool:
                attn = attn.masked_fill(~mask, torch.finfo(attn.dtype).min)
            else:
                attn = attn + mask
        attn = self.drop(F.softmax(attn, dim=-1))
        return self.out((attn @ v).transpose(1, 2).reshape(b, n, -1))


class RoPEEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, max_seq_len):
        super().__init__()
        self.attn = RoPEAttention(d_model, nhead, dropout, max_seq_len)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.Mish(), nn.Dropout(dropout),
                                nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.ff(self.norm2(x))


class RoPEEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, max_seq_len):
        super().__init__()
        self.layers = nn.ModuleList([RoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout, max_seq_len)
                                     for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class RoPEDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, max_seq_len):
        super().__init__()
        self.self_attn = RoPEAttention(d_model, nhead, dropout, max_seq_len)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.Mish(), nn.Dropout(dropout),
                                nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, input_pos=None, tgt_mask=None):
        tgt = tgt + self.self_attn(self.norm1(tgt), tgt_mask, input_pos)
        tgt = tgt + self.cross_attn(self.norm2(tgt), memory, memory)[0]
        return tgt + self.ff(self.norm3(tgt))


class RoPEDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, max_seq_len):
        super().__init__()
        self.layers = nn.ModuleList([RoPEDecoderLayer(d_model, nhead, dim_feedforward, dropout, max_seq_len)
                                     for _ in range(num_layers)])

    def forward(self, tgt, memory, input_pos=None, tgt_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, input_pos, tgt_mask)
        return tgt



class TransformerVAE_CLS(nn.Module):
    def __init__(self, ffp_dim=256, auto_dim=216, seq_len=1000, d_model=128, nhead=4, num_layers=3, latent_dim=64, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.ffp_projection = nn.Linear(ffp_dim, d_model//2)
        self.auto_projection = nn.Linear(auto_dim, d_model//2)
        self.feature_fusion = nn.Linear(d_model, d_model)
        self.ffp_norm = nn.LayerNorm(d_model//2)
        self.auto_norm = nn.LayerNorm(d_model//2)
        self.feature_norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.transformer_encoder = RoPEEncoder(d_model, nhead, num_layers, d_model*4, dropout, seq_len+1)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        self.fc_decoder_input = nn.Linear(latent_dim, d_model)
        self.decoder_queries = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.transformer_decoder = RoPEDecoder(d_model, nhead, num_layers, d_model*4, dropout, seq_len+1)
        self.output_ffp = nn.Sequential(nn.Linear(d_model, ffp_dim), nn.Softplus())
        self.output_auto = nn.Linear(d_model, auto_dim)

    def encode(self, ffp, auto):
        b = ffp.size(0)
        h_ffp = F.mish(self.ffp_norm(self.ffp_projection(ffp)))
        h_auto = F.mish(self.auto_norm(self.auto_projection(auto)))
        x = self.feature_norm(self.feature_fusion(torch.cat([h_ffp, h_auto], dim=-1)))
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)
        x = self.transformer_encoder(x)
        cls_out = x[:, 0, :]
        return self.fc_mu(cls_out), self.fc_logvar(cls_out)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        b = z.size(0)
        memory = self.fc_decoder_input(z).unsqueeze(1)
        tgt = self.decoder_queries.expand(b, -1, -1)
        input_pos = torch.arange(1, self.seq_len + 1, device=z.device)
        out = self.transformer_decoder(tgt, memory, input_pos=input_pos)
        return self.output_ffp(out), self.output_auto(out)

    def forward(self, ffp, auto):
        mu, logvar = self.encode(ffp, auto)
        z = self.reparameterize(mu, logvar)
        recon_ffp, recon_auto = self.decode(z)
        return recon_ffp, recon_auto, mu, logvar



class TransformerVAE_SET(nn.Module):
    def __init__(self, ffp_dim=256, auto_dim=216, seq_len=1000, d_model=128, nhead=4, num_layers=3, latent_dim=64, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.ffp_projection = nn.Linear(ffp_dim, d_model//2)
        self.auto_projection = nn.Linear(auto_dim, d_model//2)
        self.feature_fusion = nn.Linear(d_model, d_model)
        self.ffp_norm = nn.LayerNorm(d_model//2)
        self.auto_norm = nn.LayerNorm(d_model//2)
        self.feature_norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True, norm_first=True, activation=F.mish)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        self.fc_decoder_input = nn.Linear(latent_dim, d_model)
        self.decoder_queries = nn.Parameter(torch.randn(1, seq_len, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True, norm_first=True, activation=F.mish)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_ffp = nn.Sequential(nn.Linear(d_model, ffp_dim), nn.Softplus())
        self.output_auto = nn.Linear(d_model, auto_dim)

    def encode(self, ffp, auto):
        b = ffp.size(0)
        h_ffp = F.mish(self.ffp_norm(self.ffp_projection(ffp)))
        h_auto = F.mish(self.auto_norm(self.auto_projection(auto)))
        x = self.feature_fusion(torch.cat([h_ffp, h_auto], dim=-1))
        x = self.feature_norm(x)
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)
        x = self.transformer_encoder(x)
        cls_out = x[:, 0, :]
        return self.fc_mu(cls_out), self.fc_logvar(cls_out)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        b = z.size(0)
        memory = self.fc_decoder_input(z).unsqueeze(1)
        tgt = self.decoder_queries.expand(b, -1, -1)
        out = self.transformer_decoder(tgt=tgt, memory=memory)
        return self.output_ffp(out), self.output_auto(out)

    def forward(self, ffp, auto):
        mu, logvar = self.encode(ffp, auto)
        z = self.reparameterize(mu, logvar)
        recon_ffp, recon_auto = self.decode(z)
        return recon_ffp, recon_auto, mu, logvar


class VAE_DeepSets(nn.Module):
    def __init__(self, ffp_dim=256, auto_dim=216, seq_len=1000,
                 d_model=128, nhead=4, num_layers=3, latent_dim=64, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.ffp_dim = ffp_dim
        self.auto_dim = auto_dim
        in_features = ffp_dim + auto_dim
        self.element_net = nn.Sequential(nn.Linear(in_features, d_model),nn.LayerNorm(d_model), nn.Mish(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.global_net = nn.Sequential(nn.Linear(d_model, d_model),nn.LayerNorm(d_model), nn.Mish(), nn.Linear(d_model, latent_dim * 2))
        self.slots = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.2)
        self.shared_decoder = nn.Sequential(nn.Linear(latent_dim + d_model, d_model),nn.LayerNorm(d_model),nn.Mish(),nn.Linear(d_model, ffp_dim + auto_dim))

    def encode(self, ffp, auto):
        x = torch.cat([ffp, auto], dim=-1)          
        x = self.element_net(x)                     
        x_pool = x.mean(dim=1)                       
        out = self.global_net(x_pool)               
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        b = z.size(0)
        z_exp = z.unsqueeze(1).expand(-1, self.seq_len, -1)   
        slots = self.slots.expand(b, -1, -1)                
        combined = torch.cat([z_exp, slots], dim=-1)        
        out = self.shared_decoder(combined)                  
        recon_ffp = F.softplus(out[..., :self.ffp_dim])
        recon_auto = out[..., self.ffp_dim:]
        return recon_ffp, recon_auto

    def forward(self, ffp, auto):
        mu, logvar = self.encode(ffp, auto)
        z = self.reparameterize(mu, logvar)
        recon_ffp, recon_auto = self.decode(z)
        return recon_ffp, recon_auto, mu, logvar


# #### loss


def chamfer_distance(source, target):
    dist = torch.cdist(source, target, p=2, compute_mode="use_mm_for_euclid_dist_if_necessary").pow(2)
    src_to_tgt_min, _ = torch.min(dist, dim=-1)
    tgt_to_src_min, _ = torch.min(dist, dim=-2)
    chamfer_loss = torch.mean(src_to_tgt_min) + torch.mean(tgt_to_src_min)

    return chamfer_loss


def loss_function_set(recon_ffp, ffp_true, recon_auto, auto_true, mu, logvar, beta=1.0):
    ffp_loss = chamfer_distance(recon_ffp/0.7, ffp_true/0.7)
    auto_loss = chamfer_distance(recon_auto/4, auto_true/4)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()
    total = 0.4*ffp_loss + auto_loss + beta * kl
    return total, ffp_loss, auto_loss, kl

def loss_function_cls(recon_ffp, ffp_true, recon_auto, auto_true, mu, logvar, beta=1.0):
    ffp_loss = F.poisson_nll_loss(recon_ffp, ffp_true.float(), log_input=False, full=True, reduction='mean')
    auto_loss = F.mse_loss(recon_auto, auto_true.float())
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()
    total = 19*ffp_loss + auto_loss + beta * kl_loss
    return total, ffp_loss, auto_loss, kl_loss


# ## Train


class PolymerStreamDataset(IterableDataset):
    def __init__(self, ffp, auto, samples_per_epoch, seq_len=1000,
                 seed=42, batch_size=1):
        super().__init__()
        self.ffp = torch.as_tensor(ffp, dtype=torch.float32)
        self.auto = torch.as_tensor(auto, dtype=torch.float32)
        self.n_monomers = self.ffp.shape[0]
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self._epoch = mp.Value('i', 0)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)

    def _effective_samples(self, nw: int) -> int:
        unit = nw * self.batch_size
        return max(unit, (self.samples_per_epoch // unit) * unit)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        nw = wi.num_workers if wi is not None else 1
        wid = wi.id if wi is not None else 0

        per_worker = self._effective_samples(nw) // nw
        rng = np.random.default_rng(
            self.seed + 1_000_003 * self._epoch.value + wid
        )
        for _ in range(per_worker):
            yield self._sample(rng)

    def _sample(self, rng):
        k = int(rng.integers(1, 8))
        idxs = rng.choice(self.n_monomers, size=k, replace=False)
        r = rng.dirichlet(np.ones(k))
        exact = r * self.seq_len
        cnt = np.floor(exact).astype(np.int64)
        d = self.seq_len - int(cnt.sum())
        if d > 0:
            order = np.argsort(-(exact - cnt))
            cnt[order[:d]] += 1
        chain = np.repeat(idxs, cnt)
        rng.shuffle(chain)
        return self.ffp[chain], self.auto[chain]


# In[33]:


def build_stream_loaders(ffp, auto, cfg):
    train_ds = PolymerStreamDataset(
        ffp, auto,
        samples_per_epoch=cfg.samples_per_epoch_train,
        seq_len=cfg.seq_len, seed=cfg.seed,
        batch_size=cfg.bs,
    )
    val_ds = PolymerStreamDataset(
        ffp, auto,
        samples_per_epoch=cfg.samples_per_epoch_val,
        seq_len=cfg.seq_len, seed=cfg.seed + 7,
        batch_size=cfg.bs,
    )

    def _dl(ds, nw, bs, drop_last):
        return DataLoader(
            ds, batch_size=bs, num_workers=nw,
            pin_memory=True, drop_last=drop_last,
            persistent_workers=(nw > 0),
            prefetch_factor=cfg.pf if nw > 0 else None,
        )

    return (_dl(train_ds, cfg.nw, cfg.bs, True),
            _dl(val_ds, max(1, cfg.nw // 2), cfg.bs, False),
            train_ds, val_ds)


def train_stream(model, train_loader, val_loader, train_ds, val_ds,
                 epochs, lr, beta, device, model_name, loss_fn,
                 warmup_epochs=50, start_beta=0.0, patience=15,
                 weight_decay=0.01, scheduler_type="plateau",
                 use_amp=True, grad_clip=1.0,
                 ckpt_dir="./data/DL/models", log_dir="runs",
                 grad_hist_every=5):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5)
        sched_metric = True
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)
        sched_metric = False

    use_amp = bool(use_amp and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)
    amp_ctx = lambda: autocast("cuda", enabled=use_amp)

    writer = SummaryWriter(log_dir=f"{log_dir}/{model_name}_VAE")

    val_loss_window = deque(maxlen=4)
    best_val_smoothed = float("inf")

    best_val, no_improve = float("inf"), 0
    history = {k: [] for k in (
        "train_loss", "val_loss", "val_geom",
        "val_geom_ffp", "val_geom_auto",
        "val_ffp", "val_auto", "val_kl", "beta", "lr")}

    for epoch in range(1, epochs + 1):
        current_beta = (start_beta + (beta - start_beta) * (epoch - 1)
                        / max(1, warmup_epochs - 1)
                        if epoch <= warmup_epochs else beta)

        #  TRAIN 
        train_ds.set_epoch(epoch)
        val_ds.set_epoch(epoch)
        model.train()
        tr = {k: torch.zeros((), device=device) for k in
              ("loss", "ffp", "auto", "kl")}
        n_tr = torch.zeros((), device=device)

        for ffp, auto in train_loader:
            ffp = ffp.to(device, non_blocking=True)
            auto = auto.to(device, non_blocking=True)
            bs = ffp.size(0)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                recon_ffp, recon_auto, mu, logvar = model(ffp, auto)
                loss, f_loss, a_loss, kl_loss = loss_fn(
                    recon_ffp, ffp, recon_auto, auto, mu, logvar, current_beta)

            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            tr["loss"] += loss.detach() * bs
            tr["ffp"]  += f_loss.detach() * bs
            tr["auto"] += a_loss.detach() * bs
            tr["kl"]   += kl_loss.detach() * bs
            n_tr += float(bs)

        #  VALIDATION 
        model.eval()
        va = {k: torch.zeros((), device=device) for k in
              ("loss", "ffp", "auto", "kl", "geom_ffp", "geom_auto")}
        n_va = torch.zeros((), device=device)

        with torch.inference_mode(), amp_ctx():
            for ffp, auto in val_loader:
                ffp = ffp.to(device, non_blocking=True)
                auto = auto.to(device, non_blocking=True)
                bs = ffp.size(0)

                recon_ffp, recon_auto, mu, logvar = model(ffp, auto)
                loss, f_loss, a_loss, kl_loss = loss_fn(
                    recon_ffp, ffp, recon_auto, auto, mu, logvar, beta)

                va["loss"] += loss.detach() * bs
                va["ffp"]  += f_loss.detach() * bs
                va["auto"] += a_loss.detach() * bs
                va["kl"]   += kl_loss.detach() * bs
                va["geom_ffp"]  += chamfer_distance(
                    recon_ffp.float(), ffp.float()).detach() * bs
                va["geom_auto"] += chamfer_distance(
                    recon_auto.float(), auto.float()).detach() * bs
                n_va += float(bs)

        avg = lambda s: (s / n_va.clamp(min=1.0)).item()
        avg_tr = lambda s: (s / n_tr.clamp(min=1.0)).item()

        val_loss = avg(va["loss"])

        val_loss_window.append(val_loss)
        val_loss_smoothed = sum(val_loss_window) / len(val_loss_window)
        
        val_geom = avg(va["geom_ffp"]) + avg(va["geom_auto"])

        #  HISTORY 
        history["train_loss"].append(avg_tr(tr["loss"]))
        history["val_loss"].append(val_loss)
        history["val_loss_smoothed"].append(val_loss_smoothed)
        history["val_geom"].append(val_geom)
        history["val_geom_ffp"].append(avg(va["geom_ffp"]))
        history["val_geom_auto"].append(avg(va["geom_auto"]))
        history["val_ffp"].append(avg(va["ffp"]))
        history["val_auto"].append(avg(va["auto"]))
        history["val_kl"].append(avg(va["kl"]))
        history["beta"].append(current_beta)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        #  SCHEDULER 
        if sched_metric:
            scheduler.step(val_loss_smoothed)
        else:
            scheduler.step()

        #  LOGGING 
        for k, v in history.items():
            writer.add_scalar(f"Metrics/{k}", v[-1], epoch)

        writer.add_scalars("Loss/Total", 
                           {"Train": history["train_loss"][-1], 
                            "Val": history["val_loss"][-1]}, epoch)

        print(f"{model_name} | Ep {epoch:3d} | β {current_beta:.3f} | "
              f"Tr {history['train_loss'][-1]:.4f} | Val {val_loss:.4f} | Val_Sm {val_loss_smoothed:.4f} | "
              f"Geom {val_geom:.2f} | FFP {history['val_ffp'][-1]:.4f} | "
              f"Auto {history['val_auto'][-1]:.4f} | KL {history['val_kl'][-1]:.4f}")

        #  CHECKPOINT 
        if epoch > warmup_epochs:
            if val_loss_smoothed < best_val_smoothed - 1e-6:
                best_val_smoothed, no_improve = val_loss_smoothed, 0
                best_val = val_loss
                Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_smoothed": best_val_smoothed,
                    "best_val": best_val,
                }, f"{ckpt_dir}/{model_name}_VAE.pt")
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"Early stop @ {epoch}")
                break

            if os.path.exists(os.path.join(ckpt_dir, "STOP_TRAINING")):
                print(f"stop signal detected at epoch {epoch}.")
                break

    writer.close()
    return history


model_config = {
    'ffp_dim': 256, 'auto_dim': 216, 'seq_len': 1000,
    'd_model': 128, 'nhead': 4, 'num_layers': 3,
    'latent_dim': 64,
    'dropout': 0.12,
}

train_config = {
    'epochs': 1000000,
    'lr': 1e-4,
    'weight_decay': 0.01,
    'beta': 1,
    'warmup_epochs': 32,
    'start_beta': 0.0,
    'patience': 32,
    'scheduler_type': 'plateau',
    'grad_clip': 1.0,
}

stream_config = {
    'samples_per_epoch_train': 262144,
    'samples_per_epoch_val': 65536,
    'bs': 256,
    'nw': 8,
    'pf': 4,
    'seed': 42,
    'ckpt_dir': './data/DL/models',
    'log_dir': 'runs',
}


@dataclass
class StreamCfg:
    seq_len: int = model_config['seq_len']
    samples_per_epoch_train: int = stream_config['samples_per_epoch_train']
    samples_per_epoch_val: int = stream_config['samples_per_epoch_val']
    bs: int = stream_config['bs']
    nw: int = stream_config['nw']
    pf: int = stream_config['pf']
    epochs: int = train_config['epochs']
    lr: float = train_config['lr']
    weight_decay: float = train_config['weight_decay']
    beta: float = train_config['beta']
    warmup_epochs: int = train_config['warmup_epochs']
    start_beta: float = train_config['start_beta']
    patience: int = train_config['patience']
    seed: int = stream_config['seed']
    scheduler_type: str = train_config['scheduler_type']
    grad_clip: float = train_config['grad_clip']
    ckpt_dir: str = stream_config['ckpt_dir']
    log_dir: str = stream_config['log_dir']


if __name__ == '__main__':

    cfg = StreamCfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_loader, val_loader, train_ds, val_ds = build_stream_loaders(
        ffp_arr, auto_arr, cfg)

    model_factories = {
        'CLS': lambda: TransformerVAE_CLS(**model_config),
        'SET': lambda: TransformerVAE_SET(**model_config),
        'DS':  lambda: VAE_DeepSets(**model_config),
    }
    loss_fns = {
        'CLS': loss_function_cls,
        'SET': loss_function_set,
        'DS':  loss_function_set,
    }

    histories = {}
    for name, create_model in model_factories.items():
        print(f"\n{'='*50}\nTraining {name}\n{'='*50}")
        model = create_model().to(device)
        train_ds.set_epoch(0)

        histories[name] = train_stream(
            model, train_loader, val_loader, train_ds, val_ds,
            epochs=cfg.epochs, lr=cfg.lr, beta=cfg.beta,
            device=device, model_name=name, loss_fn=loss_fns[name],
            warmup_epochs=cfg.warmup_epochs,
            start_beta=cfg.start_beta, patience=cfg.patience,
            weight_decay=cfg.weight_decay,
            scheduler_type=cfg.scheduler_type,
            grad_clip=cfg.grad_clip,
        )

        del model
        torch.cuda.empty_cache()
        gc.collect()
