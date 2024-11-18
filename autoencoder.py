import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784):
        super(Autoencoder, self).__init__()
        
        # 인코더 레이어
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        
        # 디코더 레이어
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 인코딩
        encoded = self.encoder(x)
        # 디코딩
        decoded = self.decoder(encoded)
        return decoded

# 모델 사용 예시
if __name__ == "__main__":
    # 입력 차원은 MNIST 데이터셋 기준 (28x28 = 784)
    model = Autoencoder(input_dim=784)
    
    # 임의의 입력 데이터 
    sample_input = torch.randn(1, 784)
    
    # 모델 실행
    output = model(sample_input)
    print(f"입력 shape: {sample_input.shape}")
    print(f"출력 shape: {output.shape}")
