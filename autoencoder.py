import torch
import torch.nn as nn
import math

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

def calculate_psnr(original, reconstructed, max_pixel=1.0):
    """
    PSNR 계산 함수
    Args:
        original: 원본 데이터
        reconstructed: 복원된 데이터
        max_pixel: 픽셀의 최대값 (정규화된 데이터의 경우 1.0)
    """
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_pixel) - 10 * math.log10(mse)

# 모델 사용 예시
if __name__ == "__main__":
    # 입력 차원은 MNIST 데이터셋 기준 (28x28 = 784)
    model = Autoencoder(input_dim=784)
    
    # 임의의 입력 데이터 (0~1 사이로 정규화)
    sample_input = torch.rand(1, 784)  # randn 대신 rand 사용
    
    # MSE 손실 함수 정의
    criterion = nn.MSELoss()
    
    # 모델 실행
    output = model(sample_input)
    
    # 복원 오차 계산 (MSE)
    reconstruction_loss = criterion(output, sample_input)
    
    # PSNR 계산
    psnr_value = calculate_psnr(sample_input, output)
    
    print(f"입력 shape: {sample_input.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"복원 오차 (MSE): {reconstruction_loss.item():.6f}")
    print(f"PSNR: {psnr_value:.2f} dB")
