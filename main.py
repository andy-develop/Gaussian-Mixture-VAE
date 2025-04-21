import torch
from torchvision import datasets, transforms
from modules.vae import VAE  # 假设你的VAE类在vae.py中

def main():
    # 参数设置
    batch_size = 64
    input_size = 784  # MNIST图像展平后的维度
    hidden_size = 16
    n_classes = 10  # MNIST有10个类别
    epochs = 5
    
    # 数据加载
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_size=input_size, hidden_size=hidden_size, n_classes=n_classes).to(device)
    
    # 训练测试
    print("开始训练测试...")
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, input_size).to(device)
            recon_batch, mu, logvar = model(data)
            
            # 这里应该有损失计算和优化步骤
            # loss = model.loss_function(recon_batch, data, mu, logvar)
            # loss.backward()
            # optimizer.step() 
            
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]")
    
    # 生成测试
    print("\n开始生成测试...")
    with torch.no_grad():
        sample = torch.randn(16, hidden_size).to(device)
        generated = model.decode(sample).cpu()
        print(f"生成样本形状: {generated.shape}")  # 应该是[16, 784]

if __name__ == "__main__":
    main()
