#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",                 \
                   cudaGetErrorString(err), err, __FILE__, __LINE__);        \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

static void initRandom(std::vector<float>& v) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& x : v) {
    x = dist(rng);
  }
}

int main() {
  const int n = 1 << 24;
  const size_t bytes = n * sizeof(float);

  std::vector<float> h_a(n), h_b(n), h_c(n), h_ref(n);
  initRandom(h_a);
  initRandom(h_b);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    h_ref[i] = h_a[i] + h_b[i];
  }

  double max_err = 0.0;
  for (int i = 0; i < n; ++i) {
    double err = std::fabs(h_c[i] - h_ref[i]);
    if (err > max_err) {
      max_err = err;
    }
  }

  std::printf("VectorAdd: n=%d, time=%.3f ms, max_err=%e\n", n, ms, max_err);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}
