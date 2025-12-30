import numpy as np
import torch

class MaxMinScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def fit(self, data):
        data_o = np.array(data)
        self.max = data_o.max()
        self.min = data_o.min()

    def transform(self, data):
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        return (data - min) / (max - min + 1e-12)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min

def get_point_pred_km(y_pred, y_range):
    if y_pred.size(1) >= 6:
        y_med = y_pred[:, 4:6]   
        return y_med * y_range * 111
    else:
        lon_mid = (y_pred[:, 0] + y_pred[:, 1]) / 2
        lat_mid = (y_pred[:, 2] + y_pred[:, 3]) / 2
        mid = torch.stack([lon_mid, lat_mid], dim=1)
        return mid * y_range * 111

#point error
def huber_point_loss(y, y_range, y_pred, delta=1.0):
    y_true_km = y * y_range * 111
    y_point_km = get_point_pred_km(y_pred, y_range)

    diff = y_true_km - y_point_km
    abs_diff = torch.abs(diff)

    huber = torch.where(
        abs_diff < delta,
        0.5 * (diff ** 2) / delta,
        abs_diff - 0.5 * delta
    )
    return huber.sum(dim=1).mean()

def get_mselist(y, y_pred, y_range):
    y = y * y_range
    y_pred = y_pred * y_range
    mse = (((y - y_pred) * 111) ** 2).sum(dim=1)
    return mse

def mse_loss(y, y_pred, y_range):
    y = y * y_range
    y_pred = y_pred * y_range
    mse_list = (((y - y_pred) * 111) ** 2).sum(dim=1)
    loss = mse_list.mean()    
    return loss

#Interval loss
def pinball_loss(y, y_pred, y_range, quantiles=[0.1, 0.9], lambda_len=1):
    y = y * y_range * 111
    y_pred_lon = y_pred[:, :2] * y_range[:, 0:1] * 111
    y_pred_lat = y_pred[:, 2:] * y_range[:, 1:2] * 111
    ## L_QR
    loss = 0.0
    for dim in [0, 1]:  # 0: lon, 1: lat
        for i, q in enumerate(quantiles):
            errors = y[:, dim] - (y_pred_lon if dim == 0 else y_pred_lat)[:, i]
            loss += torch.max((q - 1) * errors, q * errors).mean()
    ## L_STR
    interval_len_lon = (y_pred_lon[:, 1] - y_pred_lon[:, 0]).clamp(min=0.0)
    interval_len_lat = (y_pred_lat[:, 1] - y_pred_lat[:, 0]).clamp(min=0.0)
    interval_len = interval_len_lon.mean() + interval_len_lat.mean()
    loss += lambda_len * interval_len
    
    return loss

def bernoulli_loss(r, att):
    loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()
    return loss

def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))

# Calibration   
def compute_q_hat(model, calibration_loader, alpha, device=0):
    model.eval()
    lon_scores = []
    lat_scores = []
    with torch.no_grad():
        for batch in calibration_loader:
            batch = batch.cuda(device)
            y = batch.y[batch.tg_mask == 1]
            y_range = batch.y_max[batch.tg_mask == 1] - batch.y_min[batch.tg_mask == 1]
            y_pred = model(batch.x, batch.edge_index, batch.tg_mask)
    
            y = y * y_range * 111
            y_pred_lon = y_pred[:, :2] * y_range[:, 0:1] * 111
            y_pred_lat = y_pred[:, 2:] * y_range[:, 1:2] * 111
            
            q_low_lon = y_pred_lon[:, 0]
            q_high_lon = y_pred_lon[:, 1]
            score_lon = torch.max(q_low_lon - y[:, 0], y[:, 0] - q_high_lon)  # for ours
            lon_scores.append(score_lon)
            
            q_low_lat = y_pred_lat[:, 0]
            q_high_lat = y_pred_lat[:, 1]
            score_lat = torch.max(q_low_lat - y[:, 1], y[:, 1] - q_high_lat)  # for ours
            lat_scores.append(score_lat)
            
    lon_scores = torch.cat(lon_scores, dim=0)
    q_hat_lon = torch.quantile(lon_scores, 1 - alpha)
    
    lat_scores = torch.cat(lat_scores, dim=0)
    q_hat_lat = torch.quantile(lat_scores, 1 - alpha)
    return q_hat_lon.item(), q_hat_lat.item()

def evaluate_metrics(model, loader, q_hat_lon, q_hat_lat, device=0):
    model.eval()
    total = 0
    covered_lon, covered_lat = 0, 0
    total_len_lon, total_len_lat = 0.0, 0.0
    total_len_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.cuda(device)
            y = batch.y[batch.tg_mask == 1]
            y_range = batch.y_max[batch.tg_mask == 1] - batch.y_min[batch.tg_mask == 1]
            y_pred = model(batch.x, batch.edge_index, batch.tg_mask)

            y = y * y_range * 111
            y_pred_lon = y_pred[:, :2] * y_range[:, 0:1] * 111
            y_pred_lat = y_pred[:, 2:] * y_range[:, 1:2] * 111

            q_low_lon = y_pred_lon[:, 0]
            q_high_lon = y_pred_lon[:, 1]
            q_low_lat = y_pred_lat[:, 0]
            q_high_lat = y_pred_lat[:, 1]
            
            q_low_lon = y_pred_lon[:, 0] - q_hat_lon
            q_high_lon = y_pred_lon[:, 1] + q_hat_lon  # for ours

            q_low_lat = y_pred_lat[:, 0] - q_hat_lat
            q_high_lat = y_pred_lat[:, 1] + q_hat_lat  # for ours
           
            # coverage
            in_interval_lon = (y[:, 0] >= q_low_lon) & (y[:, 0] <= q_high_lon)
            in_interval_lat = (y[:, 1] >= q_low_lat) & (y[:, 1] <= q_high_lat)
            covered_lon += in_interval_lon.sum().item()
            covered_lat += in_interval_lat.sum().item()

            # interval length
            total_len_lon += (q_high_lon - q_low_lon).sum().item()
            total_len_lat += (q_high_lat - q_low_lat).sum().item()

            total += y.size(0)

            for i in range(q_high_lon.size(0)):
                sample_len = ((q_high_lon[i] - q_low_lon[i]).sum().item() + 
                            (q_high_lat[i] - q_low_lat[i]).sum().item()) / 2
                total_len_list.append(sample_len)

    coverage_lon = covered_lon / total
    coverage_lat = covered_lat / total
    avg_length_lon = total_len_lon / total
    avg_length_lat = total_len_lat / total

    return {
        'coverage_lon': coverage_lon,
        'coverage_lat': coverage_lat,
        'length_lon': avg_length_lon,
        'length_lat': avg_length_lat,
        'total_len_list': total_len_list
    }

def dis_loss(y, y_pred, max, min):
    y[:, 0] = y[:, 0] * (max[0] - min[0])
    y[:, 1] = y[:, 1] * (max[1] - min[1])
    y_pred[:, 0] = y_pred[:, 0] * (max[0] - min[0])
    y_pred[:, 1] = y_pred[:, 1] * (max[1] - min[1])
    distance = torch.sqrt((((y - y_pred) * 100) * ((y - y_pred) * 100)).sum(dim=1))
    return distance

def evaluate_point_error(model, loader, device=0):
    """
    Compute MAE、RMSE、Median Distance
       mid = (low+high)/2 + delta
    """
    model.eval()
    total = 0
    total_mae, total_mse = 0.0, 0.0
    all_distances = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.cuda(device)
            y = batch.y[batch.tg_mask == 1]
            y_range = batch.y_max[batch.tg_mask == 1] - batch.y_min[batch.tg_mask == 1]
            y_pred = model(batch.x, batch.edge_index, batch.tg_mask)

            # ground truth (km)
            y_true = y * y_range * 111.0

            if y_pred.size(1) >= 6:
                mid_base_lon = (y_pred[:, 0] + y_pred[:, 1]) / 2.0
                mid_base_lat = (y_pred[:, 2] + y_pred[:, 3]) / 2.0
                mid_base = torch.stack([mid_base_lon, mid_base_lat], dim=1)  # [N,2]
                
                # delta residual (normalized)
                delta_norm = y_pred[:, 4:6]
                # res * y_range * 111
                delta_km = delta_norm * y_range * 111.0
                # mid = base_mid + delta
                y_pred_mid = (mid_base * y_range * 111.0) + delta_km

            else:
                y_pred_lon = y_pred[:, :2] * y_range[:, 0:1] * 111.0
                y_pred_lat = y_pred[:, 2:] * y_range[:, 1:2] * 111.0
                y_pred_mid = torch.cat([
                    (y_pred_lon[:, 0:1] + y_pred_lon[:, 1:2]) / 2.0,
                    (y_pred_lat[:, 0:1] + y_pred_lat[:, 1:2]) / 2.0
                ], dim=1)

            # distance (km)
            distance = torch.norm(y_true - y_pred_mid, dim=1)

            total_mae += distance.sum().item()
            total_mse += (distance ** 2).sum().item()
            total += y.size(0)
            all_distances.extend(distance.cpu().numpy())
    mae = total_mae / total
    rmse = np.sqrt(total_mse / total)
    median_dist = np.median(all_distances)
    return {"MAE": mae, "RMSE": rmse, "Median": median_dist}