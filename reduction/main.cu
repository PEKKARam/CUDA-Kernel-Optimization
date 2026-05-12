#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <numeric>
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

__global__ void reduceSum(const float* in, float* out, int n) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  float sum = 0.0f;
  if (i < n) sum += in[i];
  if (i + blockDim.x < n) sum += in[i + blockDim.x];

  sdata[tid] = sum;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[blockIdx.x] = sdata[0];
  }
}

static void initRandom(std::vector<float>& v) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (float& x : v) {
    x = dist(rng);
  }
}

static float reduceOnGpu(const float* d_in, int n) {
  const int threads = 256;
  int blocks = (n + threads * 2 - 1) / (threads * 2);

  float* d_out1 = nullptr;
  float* d_out2 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out1, blocks * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out2, blocks * sizeof(float)));

  int current_n = n;
  const float* d_current_in = d_in;
  float* d_current_out = d_out1;

  while (true) {
    blocks = (current_n + threads * 2 - 1) / (threads * 2);
    reduceSum<<<blocks, threads, threads * sizeof(float)>>>(
        d_current_in, d_current_out, current_n);
    CUDA_CHECK(cudaGetLastError());

    if (blocks == 1) {
      break;
    }

    current_n = blocks;
    d_current_in = d_current_out;
    d_current_out = (d_current_out == d_out1) ? d_out2 : d_out1;
  }

  float result = 0.0f;
  CUDA_CHECK(cudaMemcpy(&result, d_current_out, sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out1));
  CUDA_CHECK(cudaFree(d_out2));
  return result;
}

int main() {
  const int n = 1 << 24;
  const size_t bytes = n * sizeof(float);

  std::vector<float> h_in(n);
  initRandom(h_in);

  float* d_in = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  float gpu_sum = reduceOnGpu(d_in, n);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  double cpu_sum = std::accumulate(h_in.begin(), h_in.end(), 0.0);
  double err = std::fabs(cpu_sum - gpu_sum);

  std::printf("Reduction: n=%d, time=%.3f ms, cpu=%f, gpu=%f, err=%e\n",
              n, ms, cpu_sum, gpu_sum, err);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}
