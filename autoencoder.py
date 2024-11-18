import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm

class ImprovedAutoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super(ImprovedAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def calculate_metrics(original, reconstructed):
    # MSE
    mse = torch.mean((original - reconstructed) ** 2).item()
    
    # PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * math.log10(1.0) - 10 * math.log10(mse)
    
    return mse, psnr

def visualize_results(model, test_loader, device, num_images=10):
    model.eval()
    with torch.no_grad():
        # 테스트 이미지 가져오기
        images, _ = next(iter(test_loader))
        images = images[:num_images].to(device)
        
        # 재구성
        reconstructed = model(images)
        
        # 시각화
        plt.figure(figsize=(20, 4))
        for i in range(num_images):
            # 원본 이미지
            plt.subplot(2, num_images, i + 1)
            plt.imshow(images[i].cpu().view(28, 28), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('원본 이미지')
            
            # 재구성된 이미지
            plt.subplot(2, num_images, num_images + i + 1)
            plt.imshow(reconstructed[i].cpu().view(28, 28), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('재구성된 이미지')
        
        plt.tight_layout()
        plt.show()

def visualize_latent_space(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        # 잠재 공간 데이터 수집
        latent_vectors = []
        labels = []
        
        for images, targets in test_loader:
            images = images.to(device)
            latent = model.encode(images)
            latent_vectors.append(latent.cpu())
            labels.append(targets)
            
        latent_vectors = torch.cat(latent_vectors, dim=0)
        labels = torch.cat(labels, dim=0)
        
        # t-SNE 차원 축소
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_vectors[:1000].numpy())
        
        # 시각화
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                            c=labels[:1000], cmap='tab10')
        plt.colorbar(scatter)
        plt.title('t-SNE로 시각화한 잠재 공간')
        plt.show()

def train_and_evaluate(batch_size=128, num_epochs=50, latent_dim=32, learning_rate=0.001):
    # 데이터 로드 및 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                 transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, 
                                transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    
    # 모델 초기화
    model = ImprovedAutoencoder(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    early_stopping = EarlyStopping(patience=10)
    
    # 학습 결과 저장
    history = {
        'train_loss': [], 'test_loss': [],
        'train_psnr': [], 'test_psnr': []
    }
    
    # 학습 시작
    for epoch in range(num_epochs):
        # 학습
        model.train()
        train_loss = 0
        train_psnr = 0
        
        for data, _ in tqdm(train_loader, desc=f'에폭 {epoch+1}/{num_epochs}'):
            data = data.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            
            loss.backward()
            optimizer.step()
            
            mse, psnr = calculate_metrics(data, output)
            train_loss += mse
            train_psnr += psnr
        
        # 테스트
        model.eval()
        test_loss = 0
        test_psnr = 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                output = model(data)
                mse, psnr = calculate_metrics(data, output)
                test_loss += mse
                test_psnr += psnr
        
        # 평균 계산
        train_loss /= len(train_loader)
        train_psnr /= len(train_loader)
        test_loss /= len(test_loader)
        test_psnr /= len(test_loader)
        
        # 결과 저장
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_psnr'].append(train_psnr)
        history['test_psnr'].append(test_psnr)
        
        print(f'\n평균 학습 손실: {train_loss:.6f}, 평균 학습 PSNR: {train_psnr:.2f}dB')
        print(f'평균 테스트 손실: {test_loss:.6f}, 평균 테스트 PSNR: {test_psnr:.2f}dB\n')
        
        # 학습률 조정 및 조기 종료
        scheduler.step(test_loss)
        early_stopping(test_loss)
        if early_stopping.early_stop:
            print("조기 종료!")
            break
    
    # 결과 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='학습')
    plt.plot(history['test_loss'], label='테스트')
    plt.title('손실 추이')
    plt.xlabel('에폭')
    plt.ylabel('MSE 손실')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_psnr'], label='학습')
    plt.plot(history['test_psnr'], label='테스트')
    plt.title('PSNR 추이')
    plt.xlabel('에폭')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 재구성 결과 시각화
    visualize_results(model, test_loader, device)
    
    # 잠재 공간 시각화
    visualize_latent_space(model, test_loader, device)
    
    return model, history

if __name__ == "__main__":
    model, history = train_and_evaluate()
