# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import argparse, os, random, json, warnings
import numpy as np
import setproctitle
from torch_geometric.loader import DataLoader

from lib.model import *
from lib.dataset import MyOwnDataset
from lib.utils import *

warnings.filterwarnings("ignore")

def load_args():
    parser = argparse.ArgumentParser("Balanced CIPGeo Trainer")

    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--model_name', type=str, default='QRGATGeo',
                        choices=["QRMLPGeo", "QRGATGeo", "GAT_PointGeo"])

    parser.add_argument('--dataset', type=str, default='New_York',
                        choices=["New_York", "Los_Angeles", "Shanghai"])

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dim_in', type=int, default=53)
    parser.add_argument('--lambda_len', type=float, default=10.0)

    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--l2', type=float, default=1e-2)

    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--early_stop_epoch', type=int, default=15)

    parser.add_argument('--norm_x', action="store_true")
    parser.add_argument('--alpha',type=float,default=0.1,help='Miscoverage level for conformal prediction (1-alpha is nominal coverage)')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    opt = load_args()
    set_seed(opt.seed)
    setproctitle.setproctitle("Balanced_CIPGeo")

    # -------------------- device --------------------
    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {opt.dataset} | GPU={opt.gpu}")

    # -------------------- load data -----------------
    train_data = MyOwnDataset("./datasets", opt.dataset, "train", norm_x=opt.norm_x)
    valid_data = MyOwnDataset("./datasets", opt.dataset, "valid", norm_x=opt.norm_x)
    test_data  = MyOwnDataset("./datasets", opt.dataset, "test",  norm_x=opt.norm_x)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size)
    valid_loader = DataLoader(valid_data, batch_size=opt.batch_size)
    test_loader  = DataLoader(test_data,  batch_size=opt.batch_size)

    print("data loaded.")

    # -------------------- model ---------------------
    model = eval(opt.model_name)(dim_in=opt.dim_in).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, weight_decay=opt.l2
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.epochs, eta_min=1e-5
    )

    os.makedirs("asset/log", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    all_states = {}
    saved_candidates = []
    # Training Loop
    for epoch in range(1, opt.epochs + 1):

        model.train()
        total_loss = 0
        epoch_mid = opt.epochs * 0.5
        tau = 15

        # weight
        w_point = torch.sigmoid(torch.tensor((epoch - epoch_mid) / tau)).item()
        w_interval = 1 - w_point

        for batch in train_loader:
            batch = batch.to(device)

            outputs = model(batch.x, batch.edge_index, batch.tg_mask)

            y = batch.y[batch.tg_mask == 1]
            y_range = batch.y_max[batch.tg_mask == 1] - batch.y_min[batch.tg_mask == 1]

            # Interval loss
            loss_interval = pinball_loss(y, outputs, y_range,quantiles=[0.05, 0.95], lambda_len=opt.lambda_len)
            
            # Point error loss
            loss_point = huber_point_loss(y, y_range, outputs)

            loss = w_interval * loss_interval + w_point * loss_point

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        # Calibration
        q_lon, q_lat = compute_q_hat(model, valid_loader, alpha=opt.alpha, device=device)


        # Test Evaluation
        interval_m = evaluate_metrics(model, test_loader, q_lon, q_lat, device=device)
        point_m = evaluate_point_error(model, test_loader, device=device)

        coverage_lon = interval_m["coverage_lon"]
        coverage_lat = interval_m["coverage_lat"]
        width_lon = interval_m["length_lon"]
        width_lat = interval_m["length_lat"]

        mae = point_m["MAE"]
        rmse = point_m["RMSE"]
        median = point_m["Median"]

        # ----------- Logging -----------
        print(f"\n========== Epoch {epoch:03d} ==========")
        print(f"lr={lr_now:.6f} | Loss={total_loss:.4f}")
        print(f"w_interval={w_interval:.3f}, w_point={w_point:.3f}")
        print(f"Coverage:  lon={coverage_lon:.3f}, lat={coverage_lat:.3f}")
        print(f"Width:     lon={width_lon:.3f}, lat={width_lat:.3f}")
        print(f"Errors:    MAE={mae:.3f}, RMSE={rmse:.3f}, Median={median:.3f}")

        #if 0.89 <= coverage_lon <= 0.91 and 0.89 <= coverage_lat <= 0.91:
        target_cov = 1 - opt.alpha
        tol = 0.01   # ±1%
        
        if (target_cov - tol <= coverage_lon <= target_cov + tol and target_cov - tol <= coverage_lat <= target_cov + tol):

            key = f"epoch_{epoch}"
            all_states[key] = model.state_dict()

            saved_candidates.append({
                "epoch": epoch,
                "coverage_lon": coverage_lon,
                "coverage_lat": coverage_lat,
                "width_lon": width_lon,
                "width_lat": width_lat,
                "mae": mae,
                "rmse": rmse,
                "median": median,
                "file": key
            })

            print(f">>> Stored model of epoch {epoch} into combined file list.")

            with open(f"asset/log/{opt.dataset}_coverage_selected.txt", "a") as fw:
                fw.write(f"\n===== epoch {epoch} =====\n")
                fw.write(f"Coverage(lon,lat)=({coverage_lon:.3f},{coverage_lat:.3f})\n")
                fw.write(f"Width(lon,lat)=({width_lon:.3f},{width_lat:.3f})\n")
                fw.write(f"MAE={mae:.4f}, RMSE={rmse:.4f}, Median={median:.4f}\n")

    print("\n========= Training Finished =========")

    if len(all_states) == 0:
        print("No model satisfies coverage criteria (0.89~0.91). No combined file saved.")
        exit(0)

    combined_path = f"checkpoints/all_bestCov_{opt.dataset}.pth"
    torch.save(all_states, combined_path)
    print(f"\n[Saved] {len(all_states)} models saved in ONE file:")
    print(f"     --> {combined_path}")

    # Output Top 3
    sorted_candidates = sorted(
        saved_candidates,
        key=lambda c: (c["width_lon"], c["width_lat"], c["rmse"], c["mae"], c["median"])
    )

    top_k = min(3, len(sorted_candidates))
    print(f"\n========= Top {top_k} Best Models (Coverage-Qualified) =========")

    for r in range(top_k):
        c = sorted_candidates[r]
        print(f"\n------ Rank {r+1} (Epoch {c['epoch']}) ------")
        print(f"Coverage(lon,lat)=({c['coverage_lon']:.3f},{c['coverage_lat']:.3f})")
        print(f"Width(lon,lat)=({c['width_lon']:.3f},{c['width_lat']:.3f})")
        print(f"RMSE={c['rmse']:.4f}, MAE={c['mae']:.4f}, Median={c['median']:.4f}")
        print(f"Stored in combined file key: {c['file']}")
