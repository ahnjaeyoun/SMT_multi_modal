import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, confusion_matrix
import shap

# 경로 설정 (스크립트 위치 기반 상대 경로)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'sensor_features.csv')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} 파일이 없습니다. 먼저 데이터 추출 스크립트를 실행해주세요.")
        return

    print("데이터 로딩 중...")
    df = pd.read_csv(DATA_PATH)
    
    # 분석에 사용할 피처 선택
    feature_cols = [c for c in df.columns if c not in ['filename', 'label']]
    X = df[feature_cols]
    y = df['label']

    print(f"전체 데이터 개수: {len(df)}")
    print(f"라벨 분포: 정상(0) {sum(y==0)}건, 불량(1) {sum(y==1)}건")

    # 1. EDA: 상관관계 분석 및 히트맵 저장
    print("\n--- EDA 시작 ---")
    plt.figure(figsize=(12, 10))
    corr = df[feature_cols + ['label']].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'))
    plt.close()

    # 2. ML 기반 모델 학습
    print("\n--- 머신러닝 학습 및 가설 검증 시작 ---")
    # Train/Test 8:2 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # RandomForest 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 3. Threshold 최적화
    # 불량(1)일 확률값 추출
    y_scores = model.predict_proba(X_test)[:, 1]
    
    # Precision-Recall 곡선을 통해 F1-score가 최대가 되는 최적의 Threshold 계산
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10) # 0으로 나누는 것 방지
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    print(f"최적의 Threshold: {best_threshold:.4f} (이 값 이상이면 불량으로 간주)")
    print(f"최고 F1-score: {best_f1:.4f}")
    
    # 최적의 Threshold로 예측 결과 생성
    y_pred = (y_scores >= best_threshold).astype(int)
    
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred))
    
    print("[Confusion Matrix]")
    print(confusion_matrix(y_test, y_pred))
    
    # 4. SHAP (설명 가능한 AI) 분석
    print("\n--- SHAP 분석 시작 ---")
    # SHAP explainer 객체 생성 (TreeExplainer는 RF/XGB에 최적화)
    explainer = shap.TreeExplainer(model)
    # SHAP 값 계산 (시각화를 위해 X_test 샘플 데이터 사용)
    shap_values = explainer.shap_values(X_test)
    
    # Random Forest의 shap_values는 리스트 형태 [class_0_shap, class_1_shap] 인 경우가 많음
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1] # 불량(1) 클래스에 대한 SHAP 값
    else:
        shap_values_to_plot = shap_values

    # a. Summary Plot: 모델이 전체적으로 어떤 피처를 가장 중요하게 봤는지
    plt.figure()
    shap.summary_plot(shap_values_to_plot, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'shap_summary_plot.png'))
    plt.close()
    
    # b. Local Explanation (Force Plot / Waterfall - 특정 샘플 해석)
    # y_test 중에 불량(1)으로 제대로 예측한 인덱스 하나 추출
    defect_indices = np.where((y_test == 1) & (y_pred == 1))[0]
    if len(defect_indices) > 0:
        idx = defect_indices[0] # 첫 번째 케이스
        
        # 최신 shap의 waterfall plot을 위해 Explanation 객체 생성
        exp = shap.Explanation(values=shap_values_to_plot[idx], 
                               base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value, 
                               data=X_test.iloc[idx].values, 
                               feature_names=X_test.columns)
        
        plt.figure()
        shap.waterfall_plot(exp, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'shap_waterfall_defect_case_{idx}.png'))
        plt.close()
        print(f"SHAP Waterfall Plot 저장 완료: (인덱스 {idx} 불량 케이스 분석)")

    print(f"\n모든 시각화 결과는 {RESULTS_DIR} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
