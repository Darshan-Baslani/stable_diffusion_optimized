# Stable Diffusion in PyTorch

The goal of this project is to implement Stable Diffusion in PyTorch from scratch, and then to optimize the inference time. This project is a learning exercise in inference optimization.

## Performance

The standard inference time for a 512x512 image with 50 inference steps on a Colab T4 GPU was 33 seconds.

### Optimizations

-   **Flash Attention**: By implementing Flash Attention, the inference time was reduced to 26 seconds, which is a **21.2%** improvement.

### Future Improvements

Future optimizations are planned to further reduce inference time.
