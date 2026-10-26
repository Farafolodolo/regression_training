"""
models.py
Definiciones de arquitecturas neuronales en PyTorch para el proyecto Nano-AI (Semana 1).
Cumple estrictamente la regla de "Random Starting Weights" (entrenamiento desde cero sin
pesos preentrenados) mediante inicialización explícita Kaiming He (capas ReLU) y Xavier Glorot
(capas de salida lineal).

Modelos incluidos:
1. ForecasterRegressor (Track 4): MLP 20 -> 32 -> 16 -> 1
2. AnomalyAutoencoder (Track 3): Autoencoder simétrico 12 -> 8 -> 3 (bottleneck) -> 8 -> 12

Autor: David Isaí Portillo López (R2 - Model Lead)
"""

from __future__ import annotations

from typing import Dict, Any
import torch
import torch.nn as nn


class ForecasterRegressor(nn.Module):
    """
    Modelo 1 (Track 4): Red Neuronal Perceptrón Multicapa (MLP) para predicción térmica continua (+30 min).
    Topología compacta: 20 -> 32 (ReLU) -> 16 (ReLU) -> 1 (Lineal).
    """

    def __init__(self, input_dim: int = 20, output_dim: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.act2 = nn.ReLU()
        self.fc_out = nn.Linear(16, output_dim)

        self._init_random_weights()

    def _init_random_weights(self) -> None:
        """
        Inicialización explícita de pesos aleatorios desde cero:
        - Kaiming Normal (He) para capas ocultas con activación ReLU.
        - Xavier Uniform (Glorot) para la capa de regresión lineal de salida.
        """
        for layer in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        out = self.fc_out(x)
        return out


class AnomalyAutoencoder(nn.Module):
    """
    Modelo 2 (Track 3): Autoencoder simétrico con cuello de botella para detección no supervisada
    de anomalías térmicas mediante Error Cuadrático Medio (MSE) de reconstrucción.
    Topología en espejo: 12 -> 8 (ReLU) -> 3 (Bottleneck, ReLU) -> 8 (ReLU) -> 12 (Lineal).
    """

    def __init__(self, input_dim: int = 12, latent_dim: int = 3) -> None:
        super().__init__()
        # Encoder: compresión hacia el espacio latente
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
            nn.ReLU(),
        )
        # Decoder: reconstrucción simétrica hacia el espacio original
        self.decoder_hidden = nn.Linear(latent_dim, 8)
        self.decoder_act = nn.ReLU()
        self.decoder_out = nn.Linear(8, input_dim)

        self._init_random_weights()

    def _init_random_weights(self) -> None:
        """
        Inicialización explícita de pesos aleatorios desde cero:
        - Kaiming Normal (He) en las capas con activación ReLU (Encoder y capa oculta del Decoder).
        - Xavier Uniform (Glorot) en la capa de reconstrucción lineal final.
        """
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)

        nn.init.kaiming_normal_(self.decoder_hidden.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.decoder_hidden.bias)

        nn.init.xavier_uniform_(self.decoder_out.weight)
        nn.init.zeros_(self.decoder_out.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Proyecta la entrada de 12 dimensiones al cuello de botella latente de 3 dimensiones."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruye el vector de 12 dimensiones a partir del vector latente z."""
        h = self.decoder_act(self.decoder_hidden(z))
        return self.decoder_out(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed


def summarize_model_footprint(model: nn.Module, model_name: str = "") -> Dict[str, Any]:
    """
    Calcula e imprime el número exacto de parámetros entrenables, no entrenables,
    buffers y el peso en memoria (KB) en precisión FP32 para despliegue en Edge (Raspberry Pi).
    """
    name = model_name or model.__class__.__name__
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_kb = (param_bytes + buffer_bytes) / 1024.0

    layer_breakdown = []
    for layer_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            l_params = sum(p.numel() for p in module.parameters())
            layer_breakdown.append(
                f"{layer_name}: Linear({module.in_features} -> {module.out_features}) [{l_params} params]"
            )

    print("-" * 72)
    print(f"  Resumen de Arquitectura : {name}")
    print(f"  Parámetros Entrenables  : {trainable_params:,} (Total: {total_params:,})")
    print(f"  Tamaño en Memoria (FP32): {total_kb:.4f} KB ({param_bytes + buffer_bytes:,} bytes)")
    print("  Desglose de Capas Lineales:")
    for item in layer_breakdown:
        print(f"    - {item}")
    print("-" * 72)

    return {
        "model_name": name,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "size_bytes": param_bytes + buffer_bytes,
        "size_kb": total_kb,
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    reg_model = ForecasterRegressor()
    ae_model = AnomalyAutoencoder()

    summarize_model_footprint(reg_model, "ForecasterRegressor (Track 4)")
    summarize_model_footprint(ae_model, "AnomalyAutoencoder (Track 3)")
