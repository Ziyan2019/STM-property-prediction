import torch
from torch_geometric.data import DataLoader
from my_dimenet_molecule import DimeNet
from torch.optim import Adam
from sklearn.metrics import mean_absolute_error, r2_score  # 导入 r2_score
from dimnet_util import load_data, load_data_simple
from torch.optim.lr_scheduler import StepLR


# 检查 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_dirs = ["../stm_image_908", "../stm_image_1148"]

# csv_paths = ["../compas-1D_cam-b3lyp_aug-cc-pvdz.csv", "../compas-3D_cam-b3lyp_aug-cc-pvdz.csv"]
# target_column = 'GAP_eV'

csv_paths = ["../polar_cata_908.csv", "../polar_peri_1145.csv"]
target_column = 'polar'

# csv_paths = ["../dipole_cata_908.csv", "../dipole_peri_1145.csv"]
# target_column = 'dipole_debye'

train_data, test_data = load_data(image_dirs, csv_paths, target_column)
#train_data, test_data = load_data_simple(image_dirs, csv_paths, target_column)


# 创建 DataLoader
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# 打印 DataLoader 信息
print(f"Train DataLoader: {len(train_loader)} batches")
print(f"Test DataLoader: {len(test_loader)} batches")

model = DimeNet(
    hidden_channels=128,  # 隐藏层维度
    out_channels=1,  # 输出维度（预测 GAP_eV）
    num_blocks=6,  # 模块数量
    num_bilinear=8,  # 双线性层维度
    num_spherical=7,  # 球谐函数数量
    num_radial=6,  # 径向基函数数量
    cutoff=5,  # 截断距离
    envelope_exponent=5,  # 平滑截断指数
    num_before_skip=1,  # 跳跃连接前的残差层数
    num_after_skip=2,  # 跳跃连接后的残差层数
    num_output_layers=3,  # 输出层数
    act='swish',  # 激活函数
    output_initializer='zeros'  # 输出层初始化
).to(device)

# model = DegeneratedDimeNet(hidden_channels=64, out_channels=1, num_interactions=3, cutoff=1.7).to(device)



# 定义损失函数和优化器
criterion = torch.nn.L1Loss()  # 使用 L1 损失
# 定义优化器
optimizer = Adam(model.parameters(), lr=1e-3)
# 定义学习率调度器（每 10 个 epoch 将学习率乘以 0.1）
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

# 训练函数
def train(epoch):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.z, batch.pos, batch.batch)  # 前向传播
        # out = model(batch.pos, batch.batch) 
        # 调整 batch.y 的形状为 (batch_size, 1)
        target = batch.y.view(-1, 1)
        loss = criterion(out, target)  # 计算损失
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 验证函数
def validate():
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.z, batch.pos, batch.batch)
            # out = model(batch.pos, batch.batch)
            # 调整 batch.y 的形状为 (batch_size, 1)
            target = batch.y.view(-1, 1)
            predictions.extend(out.view(-1).cpu().numpy())
            labels.extend(target.view(-1).cpu().numpy())
    print(["{:.4f}".format(x) for x in labels])
    print(["{:.4f}".format(x) for x in predictions])
    # 计算 MAE 和 R²
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return mae, r2

# 训练循环
num_epochs = 500
best_val_mae = float('inf')
for epoch in range(1, num_epochs + 1):
    train_loss = train(epoch)
    val_mae, val_r2 = validate()  # 获取 MAE 和 R²
    print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}')
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), 'best_dimenet.pt')

# 测试函数
def test():
    model.load_state_dict(torch.load('best_dimenet.pt'))
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.z, batch.pos, batch.batch)
            # 调整 batch.y 的形状为 (batch_size, 1)
            target = batch.y.view(-1, 1)
            predictions.extend(out.view(-1).cpu().numpy())
            labels.extend(target.view(-1).cpu().numpy())
    # 计算 MAE 和 R²
    test_mae = mean_absolute_error(labels, predictions)
    test_r2 = r2_score(labels, predictions)
    print(f'Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}')

# 在测试集上评估模型
test()
