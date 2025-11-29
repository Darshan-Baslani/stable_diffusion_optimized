# Stable Diffusion in PyTorch

The goal of this project is to implement Stable Diffusion in PyTorch from scratch, and then to optimize the inference time. This project is a learning exercise in inference optimization.

## Performance

Inference is performed for a 512x512 image with 50 inference steps. The deployment platform was changed from Google Colab to [Modal](https://modal.com) to leverage more powerful GPUs.

| GPU         | Previous Time (T4) | Current Time (A10) | Improvement |
|-------------|--------------------|--------------------|-------------|
| **Time (s)**| 33s                | ~4.7s              | **~85.7%**  |


## Optimizations

### Flash Attention

We replaced the standard attention mechanism with Flash Attention, which is a more memory-efficient and faster implementation. The code conditionally uses `torch.nn.attention.SDPBackend.FLASH_ATTENTION` for smaller head dimensions and falls back to the `MATH` backend for larger ones to ensure compatibility and performance.

### Quantization

To further boost performance, we apply quantization to the model.

-   **Weights**: Quantized to `int8`.
-   **Activations**: Kept at `float16`.

This mixed-precision approach significantly speeds up computation. The quantization is implemented using the `torch.ao.quantization` module, with `QuantStub` and `DeQuantStub` marking the quantization and dequantization points in the model's forward pass.

The switch from a Google Colab T4 GPU to an A10 GPU on Modal was necessary because PyTorch's support for Flash Attention with mixed-precision inputs (like `int8` weights) is not available on the T4's Turing architecture. The A10 (Ampere architecture) provides the necessary hardware and kernel support for these optimizations.

## Deployment on Modal

The project is deployed on [Modal](https.modal.com) for simplified access to high-performance GPUs like the A10.

-   `modal_inference.py`: Defines the Modal application, specifying the container image, dependencies (`torch`, `torchao`, etc.), and the `A10` GPU requirement.
-   `modal_volume.py`: Manages the model weights. It downloads the `v1-5-pruned-emaonly.ckpt` file into a persistent Modal Volume, so it doesn't need to be downloaded on every run.
-   `modal_infer.sh`: A simple shell script that is executed inside the Modal container to run the main inference script (`naive/sd/inference.py`).