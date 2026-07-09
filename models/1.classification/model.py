"""
SMT 불량 원인 분석 모델 (model.py)

이미지와 센서 시계열을 융합하여 불량 유형(16종)과 센서 기여도(5개)를 동시에 출력한다.

모델 구조:
  이미지 → EfficientNet-B3 → 1536d
  센서   → 채널독립 PatchTST → 256d + 기여도(5,)
  Concat(1792d) → FC → 16-dim logit

사용법:
  from model import SMTCauseAnalyzer
  model  = SMTCauseAnalyzer()
  logits, attn = model(image, sensor)       # 학습 시
  result = model.predict(image, sensor)     # 추론 시
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# 센서 채널 이름 (순서가 센서 데이터의 컬럼 순서와 일치해야 함)
SENSOR_NAMES = ["temperature", "humidity", "vibration", "acceleration", "noise"]

# 불량 유형 이름 (category_id 17→index 0, 18→index 1, ..., 32→index 15)
DEFECT_NAMES = [
    "미납", "납부족", "납쇼트", "납볼", "납좌표밀림", "납형성불량",
    "냉납", "밀림", "쇼트", "오삽", "미삽", "역삽",
    "뒤집힘", "일어섬", "납금감/핀홀", "납고드름",
]


# 단일 센서 채널을 Transformer 로 인코딩
class ChannelEncoder(nn.Module):
    """
    단일 센서 채널의 시계열 (seq_len,) 을 d_model 차원 벡터로 인코딩한다.

    PatchTST 방식 적용:
      1. 시계열을 patch_len 크기의 패치로 분할
      2. 각 패치를 Linear projection → d_model 차원 임베딩
      3. CLS 토큰 + 위치 임베딩 추가
      4. Transformer Encoder 통과
      5. CLS 토큰 위치의 출력을 채널 표현 벡터로 사용

    Args:
        seq_len   : 입력 시계열 길이 (ex: 5)
        patch_len : 패치 하나의 크기 (ex: 1 → 패치 5개)
        d_model   : Transformer 내부 차원
        nhead     : Multi-head Attention 헤드 수
        num_layers: Transformer Encoder 레이어 수
        dropout   : 드롭아웃 비율
    """

    def __init__(self, seq_len, patch_len, d_model, nhead, num_layers, dropout):
        super().__init__()

        # seq_len 이 patch_len 의 배수여야 나머지 없이 패치 분할 가능
        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        num_patches = seq_len // patch_len  # ex: 5 // 1 = 5개 패치

        # 패치 하나를 d_model 차원으로 선형 투영
        self.patch_proj = nn.Linear(patch_len, d_model)

        # CLS 토큰: BERT에서 온 개념으로, 시퀀스 전체 의미를 하나의 벡터로 요약하는 역할
        # Transformer 통과 후 이 토큰 위치의 출력이 전체 시계열 표현이 된다
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 위치 임베딩: CLS(1개) + 패치(num_patches개) 각각의 위치 정보
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # 작은 값으로 초기화

        # Transformer Encoder (Pre-LN 구조로 학습 안정성 향상)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # FFN 내부 차원은 d_model 의 4배
            dropout=dropout,
            batch_first=True,   # 입력 형태: (Batch, Seq, Feature)
            norm_first=True,    # Pre-LN: LayerNorm → Attention 순서 (학습 안정)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)  # 최종 출력 정규화
        self.patch_len   = patch_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len)  — 단일 센서 채널의 시계열 배치

        Returns:
            (B, d_model)  — CLS 토큰에 집약된 채널 표현 벡터
        """
        B, T = x.shape  # B: 배치 크기(한 번에 처리하는 샘플 수), T: 시계열 길이(seq_len)

        # 시계열을 패치로 분할: (B, seq_len) → (B, num_patches, patch_len)
        x = x.reshape(B, T // self.patch_len, self.patch_len)

        # 각 패치를 d_model 차원으로 투영: (B, num_patches, d_model)
        x = self.patch_proj(x)

        # CLS 토큰을 배치 크기만큼 복제하여 시퀀스 앞에 추가
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                 # (B, num_patches+1, d_model)

        # 위치 임베딩 더하기 (위치 정보 주입)
        x = x + self.pos_embed

        # Transformer 통과
        x = self.transformer(x)

        # CLS 토큰(index=0) 위치의 출력을 채널 전체 표현으로 사용
        return self.norm(x[:, 0])   # (B, d_model)


# 5개 센서 채널을 독립 인코딩한 뒤 채널별 기여도 attention 을 산출
class SensorEncoder(nn.Module):
    """
    센서 5채널을 각각 독립적으로 인코딩한 뒤,
    채널별 attention 으로 각 센서의 불량 기여도를 계산한다.

    채널 독립(Channel-Independent) 방식의 장점:
      - 채널 간 스케일 차이(온도 20~30°C, 소음 60~80dB)에 영향받지 않음
      - 채널별 패턴을 독립적으로 학습 → 각 센서의 이상 패턴 포착에 유리
      - Attention weight 가 자연스럽게 센서 기여도로 해석 가능

    Args:
        seq_len    : 센서 시계열 길이 (5)
        n_channels : 센서 채널 수 (5)
        patch_len  : 패치 크기 (1)
        d_model    : 채널 인코더 내부 차원 (128)
        nhead      : Attention 헤드 수 (4)
        num_layers : Transformer 레이어 수 (2)
        dropout    : 드롭아웃 비율 (0.1)
        sensor_out : 최종 센서 피처 차원 (256)

    Returns (forward):
        feat        : (B, sensor_out)  — 융합된 센서 피처 벡터
        attn_weights: (B, n_channels)  — 각 센서의 기여도 (softmax, 합=1)
    """

    def __init__(
        self,
        seq_len=5, n_channels=5, patch_len=1,
        d_model=128, nhead=4, num_layers=2,
        dropout=0.1, sensor_out=256,
    ):
        super().__init__()
        self.n_channels = n_channels  # forward에서 채널 수만큼 반복하기 위해 저장

        # 채널마다 독립적인 ChannelEncoder 인스턴스 생성
        # (파라미터 공유 없음 → 각 채널의 특성을 개별 학습)
        self.channel_encoders = nn.ModuleList([
            ChannelEncoder(seq_len, patch_len, d_model, nhead, num_layers, dropout)
            for _ in range(n_channels)
        ])

        # 채널별 기여도 계산: d_model → scalar (1)
        # 이 scalar 에 softmax 를 적용하면 5개 채널의 기여도 합=1
        self.attn_proj = nn.Linear(d_model, 1)

        # 가중합된 채널 표현을 sensor_out 차원으로 투영
        self.out_proj = nn.Linear(d_model, sensor_out)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, seq_len, n_channels)  — 전체 센서 시계열

        Returns:
            feat        : (B, sensor_out)
            attn_weights: (B, n_channels)  — 센서 기여도
        """
        # 채널별 독립 인코딩
        feats = []
        for i, enc in enumerate(self.channel_encoders):
            # x[:, :, i] → i번째 채널 시계열: (B, seq_len)
            feats.append(enc(x[:, :, i]))   # (B, d_model)

        # 5개 채널 표현을 하나의 텐서로 결합: (B, 5, d_model)
        feats = torch.stack(feats, dim=1)

        # 각 채널에 대해 scalar score 계산 후 softmax → 기여도
        # attn_proj: (B, 5, d_model) → (B, 5, 1) → squeeze → (B, 5)
        scores = self.attn_proj(feats).squeeze(-1)  # (B, 5)
        attn   = F.softmax(scores, dim=-1)           # (B, 5) — 합=1, 센서 기여도

        # 기여도로 가중합: (B, 5, 1) * (B, 5, d_model) → sum → (B, d_model)
        feat = (attn.unsqueeze(-1) * feats).sum(dim=1)

        # 최종 차원 투영: (B, d_model) → (B, sensor_out)
        feat = self.out_proj(feat)

        return feat, attn


# 이미지 + 센서 융합 메인 모델
class SMTCauseAnalyzer(nn.Module):
    """
    이미지와 센서 데이터를 융합하여
    불량 유형 확률(16종)과 센서 기여도(5개)를 동시에 출력하는 메인 모델.

    Args:
        sensor_out : SensorEncoder 출력 차원 (기본값 256)
        dropout    : 분류 헤드 드롭아웃 비율 (기본값 0.3)
    """
    NUM_DEFECTS = 16  # 불량 유형 수 (category_id 17~32)

    def __init__(self, sensor_out: int = 256, dropout: float = 0.3):
        super().__init__()

        # ImageNet으로 사전학습된 EfficientNet-B3 가중치를 그대로 사용 (전이학습)
        # 마지막 분류 레이어만 Identity(아무것도 안 하는 레이어)로 교체해
        # 1536차원 특징 벡터만 꺼내 쓴다 — SMT 불량 분류는 뒤에 붙이는 FC 레이어가 담당
        self.image_encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        vision_dim = self.image_encoder.classifier[1].in_features  # 1536
        self.image_encoder.classifier = nn.Identity()

        # 센서 5채널을 각각 Transformer로 인코딩하고 채널별 기여도까지 함께 계산
        self.sensor_encoder = SensorEncoder(sensor_out=sensor_out)

        # 이미지(1536) + 센서(256) = 1792d 를 입력으로 받는 분류 헤드
        fused_dim = vision_dim + sensor_out
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),  # 1792 → 512
            nn.GELU(),                  # GELU 활성화 (ReLU보다 부드러운 기울기)
            nn.Dropout(dropout),
            nn.Linear(512, 128),        # 512 → 128
            nn.GELU(),
            nn.Dropout(dropout / 2),    # 후반 레이어는 드롭아웃 약하게
            nn.Linear(128, self.NUM_DEFECTS),  # 128 → 16 (불량 유형 수)
            # 활성화 함수 없음: 손실 함수(BCEWithLogitsLoss)가 내부적으로 sigmoid를
            # 수치적으로 안정된 방식으로 적용하므로 여기서 sigmoid를 쓰면 중복이 된다
        )

    def forward(self, image: torch.Tensor, sensor: torch.Tensor):
        """
        학습 시 사용하는 forward pass.

        Args:
            image : (B, 3, H, W)      — 전처리된 이미지 배치
            sensor: (B, seq_len, 5)   — 센서 시계열 배치

        Returns:
            logits      : (B, 16)  — 불량 유형별 logit (BCEWithLogitsLoss 입력용)
            attn_weights: (B, 5)   — 센서 기여도 (온도/습도/진동/가속도/소음, 합=1)
        """
        # 이미지 → 1536d 피처
        img_feat = self.image_encoder(image)

        # 센서 → 256d 피처 + 5개 채널 기여도
        sensor_feat, attn = self.sensor_encoder(sensor)

        # 이미지 + 센서 피처 결합 → 불량 유형 logit
        fused  = torch.cat([img_feat, sensor_feat], dim=1)   # (B, 1792)
        logits = self.classifier(fused)                       # (B, 16)

        return logits, attn

    @torch.no_grad()
    def predict(self, image: torch.Tensor, sensor: torch.Tensor, threshold: float = 0.5) -> dict:
        """
        추론 전용 메서드: 사람이 읽기 쉬운 딕셔너리 형태로 결과를 반환한다.

        Args:
            image    : (1, 3, H, W)  — 단일 이미지 (배치 크기 1)
            sensor   : (1, seq_len, 5)
            threshold: 불량으로 판정할 확률 임계값 (기본값 0.5)

        Returns:
            {
              "defect_probs"    : {"미납": 0.82, "납부족": 0.12, ...},   # 16종 확률
              "sensor_contrib"  : {"temperature": 0.08, "vibration": 0.51, ...},  # 기여도
              "detected_defects": ["미납", ...],   # threshold 이상인 불량 유형
            }

        사용 예:
            result = model.predict(image, sensor)
            print("감지된 불량:", result["detected_defects"])
            print("주요 원인 센서:", max(result["sensor_contrib"], key=result["sensor_contrib"].get))
        """
        self.eval()
        logits, attn = self.forward(image, sensor)

        # logit → 확률 (sigmoid)
        probs = torch.sigmoid(logits)[0].cpu()   # (16,)
        attn  = attn[0].cpu()                     # (5,)

        # 불량 유형별 확률 딕셔너리
        defect_probs   = {n: round(probs[i].item(), 4) for i, n in enumerate(DEFECT_NAMES)}

        # 센서 기여도 딕셔너리 (이미 softmax 적용됨)
        sensor_contrib = {n: round(attn[i].item(), 4) for i, n in enumerate(SENSOR_NAMES)}

        # threshold 이상인 불량 유형만 감지됨으로 표시
        detected = [n for n, p in defect_probs.items() if p >= threshold]

        return {
            "defect_probs":     defect_probs,
            "sensor_contrib":   sensor_contrib,
            "detected_defects": detected,
        }
