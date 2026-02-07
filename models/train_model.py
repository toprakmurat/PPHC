#!/usr/bin/env python3
"""
SecurePulse-FHE: Model Training and Quantization Pipeline

This module handles:
  1. Training plaintext ML models (Logistic Regression, XGBoost)
  2. Quantization from FP32 to INT8 for FHE compatibility
  3. Model export for encrypted inference
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


@dataclass
class QuantizationConfig:
    """Configuration for INT8 quantization."""
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = False
    calibration_samples: int = 1000


class Quantizer:
    """
    Quantizes FP32 model weights and activations to INT8.
    
    FHE schemes like CKKS/BFV operate on fixed-point or integer data.
    Quantization reduces precision loss during encrypted inference.
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.scale: Optional[float] = None
        self.zero_point: Optional[int] = None
    
    def calibrate(self, data: np.ndarray) -> None:
        """Compute quantization parameters from calibration data."""
        min_val = np.min(data)
        max_val = np.max(data)
        
        if self.config.symmetric:
            abs_max = max(abs(min_val), abs(max_val))
            self.scale = abs_max / (2 ** (self.config.bits - 1) - 1)
            self.zero_point = 0
        else:
            self.scale = (max_val - min_val) / (2 ** self.config.bits - 1)
            self.zero_point = int(round(-min_val / self.scale))
    
    def quantize(self, data: np.ndarray) -> np.ndarray:
        """Convert FP32 array to INT8."""
        if self.scale is None:
            self.calibrate(data)
        
        quantized = np.round(data / self.scale) + self.zero_point
        qmin = -(2 ** (self.config.bits - 1)) if self.config.symmetric else 0
        qmax = 2 ** (self.config.bits - 1) - 1 if self.config.symmetric else 2 ** self.config.bits - 1
        
        return np.clip(quantized, qmin, qmax).astype(np.int8)
    
    def dequantize(self, data: np.ndarray) -> np.ndarray:
        """Convert INT8 array back to FP32."""
        return (data.astype(np.float32) - self.zero_point) * self.scale


def generate_synthetic_health_data(
    n_samples: int = 10000,
    n_features: int = 12,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic health risk dataset.
    
    Features simulate common health metrics:
      - Age, BMI, blood pressure, cholesterol, glucose, etc.
    
    Target: Binary classification (0 = low risk, 1 = high risk)
    """
    rng = np.random.RandomState(random_state)
    
    # Feature generation with realistic distributions
    age = rng.normal(50, 15, n_samples).clip(18, 90)
    bmi = rng.normal(26, 5, n_samples).clip(15, 45)
    systolic_bp = rng.normal(120, 20, n_samples).clip(80, 200)
    diastolic_bp = rng.normal(80, 12, n_samples).clip(50, 120)
    cholesterol = rng.normal(200, 40, n_samples).clip(100, 350)
    glucose = rng.normal(100, 25, n_samples).clip(60, 300)
    heart_rate = rng.normal(72, 12, n_samples).clip(45, 120)
    smoking = rng.binomial(1, 0.2, n_samples)
    diabetes = rng.binomial(1, 0.1, n_samples)
    family_history = rng.binomial(1, 0.25, n_samples)
    exercise_hours = rng.exponential(3, n_samples).clip(0, 20)
    sleep_hours = rng.normal(7, 1.5, n_samples).clip(3, 12)
    
    X = np.column_stack([
        age, bmi, systolic_bp, diastolic_bp, cholesterol, glucose,
        heart_rate, smoking, diabetes, family_history, exercise_hours, sleep_hours
    ])
    
    # Risk score based on feature combinations
    risk_score = (
        0.03 * age +
        0.08 * bmi +
        0.02 * systolic_bp +
        0.01 * diastolic_bp +
        0.01 * cholesterol +
        0.015 * glucose +
        2.0 * smoking +
        3.5 * diabetes +
        1.5 * family_history -
        0.15 * exercise_hours -
        0.1 * sleep_hours
    )
    
    threshold = np.percentile(risk_score, 70)
    y = (risk_score > threshold).astype(int)
    
    return X, y


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    regularization: float = 1.0
) -> Tuple[LogisticRegression, StandardScaler]:
    """Train a logistic regression model for health risk prediction."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(
        C=regularization,
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_scaled, y_train)
    
    return model, scaler


def export_quantized_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    quantizer: Quantizer,
    output_path: Path
) -> dict:
    """
    Export quantized model parameters for FHE inference.
    
    Returns dictionary with INT8 weights and scaling factors.
    """
    weights = model.coef_.flatten()
    bias = model.intercept_
    
    quantizer.calibrate(weights)
    quantized_weights = quantizer.quantize(weights)
    
    export_data = {
        'weights_int8': quantized_weights.tolist(),
        'bias_fp32': bias.tolist(),
        'weight_scale': quantizer.scale,
        'weight_zero_point': quantizer.zero_point,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
        'quantization_bits': quantizer.config.bits
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **export_data)
    
    return export_data


def main():
    print("=" * 60)
    print("SecurePulse-FHE: Model Training Pipeline")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n[1/4] Generating synthetic health data...")
    X, y = generate_synthetic_health_data(n_samples=10000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Training samples: {len(X_train)}")
    print(f"      Test samples: {len(X_test)}")
    
    # Train model
    print("\n[2/4] Training Logistic Regression model...")
    model, scaler = train_logistic_regression(X_train, y_train)
    
    # Evaluate
    print("\n[3/4] Evaluating model performance...")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"      Accuracy: {accuracy:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    
    # Quantize and export
    print("[4/4] Quantizing model to INT8...")
    quantizer = Quantizer(QuantizationConfig(bits=8, symmetric=True))
    output_path = Path(__file__).parent / "exports" / "health_risk_model.npz"
    export_data = export_quantized_model(model, scaler, quantizer, output_path)
    
    print(f"      Quantization scale: {export_data['weight_scale']:.6f}")
    print(f"      Model exported to: {output_path}")
    print("\n" + "=" * 60)
    print("Training complete. Model ready for FHE inference.")
    print("=" * 60)


if __name__ == "__main__":
    main()
