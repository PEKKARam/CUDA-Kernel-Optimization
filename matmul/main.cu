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

static constexpr int TILE = 16;

__global__ void matmulTiled(const float* A, const float* B, float* C,
														int M, int N, int K) {
	__shared__ float sA[TILE][TILE];
	__shared__ float sB[TILE][TILE];

	int row = blockIdx.y * TILE + threadIdx.y;
	int col = blockIdx.x * TILE + threadIdx.x;

	float acc = 0.0f;
	int tiles = (K + TILE - 1) / TILE;

	for (int t = 0; t < tiles; ++t) {
		int a_col = t * TILE + threadIdx.x;
		int b_row = t * TILE + threadIdx.y;

		sA[threadIdx.y][threadIdx.x] =
				(row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
		sB[threadIdx.y][threadIdx.x] =
				(b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
		__syncthreads();

		#pragma unroll
		for (int k = 0; k < TILE; ++k) {
			acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
		}
		__syncthreads();
	}

	if (row < M && col < N) {
		C[row * N + col] = acc;
	}
}

static void initRandom(std::vector<float>& v) {
	std::mt19937 rng(123);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (float& x : v) {
		x = dist(rng);
	}
}

static void matmulCpu(const std::vector<float>& A, const std::vector<float>& B,
											std::vector<float>& C, int M, int N, int K) {
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			float acc = 0.0f;
			for (int k = 0; k < K; ++k) {
				acc += A[i * K + k] * B[k * N + j];
			}
			C[i * N + j] = acc;
		}
	}
}

int main() {
	const int M = 1024;
	const int N = 1024;
	const int K = 1024;
	const size_t bytesA = M * K * sizeof(float);
	const size_t bytesB = K * N * sizeof(float);
	const size_t bytesC = M * N * sizeof(float);

	std::vector<float> h_A(M * K);
	std::vector<float> h_B(K * N);
	std::vector<float> h_C(M * N);
	std::vector<float> h_ref(M * N);
	initRandom(h_A);
	initRandom(h_B);

	float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
	CUDA_CHECK(cudaMalloc(&d_A, bytesA));
	CUDA_CHECK(cudaMalloc(&d_B, bytesB));
	CUDA_CHECK(cudaMalloc(&d_C, bytesC));
	CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice));

	dim3 block(TILE, TILE);
	dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start));
	matmulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	CUDA_CHECK(cudaGetLastError());

	float ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

	CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost));
	matmulCpu(h_A, h_B, h_ref, M, N, K);

	double max_err = 0.0;
	for (int i = 0; i < M * N; ++i) {
		double err = std::fabs(h_C[i] - h_ref[i]);
		if (err > max_err) {
			max_err = err;
		}
	}

	double gflops = (2.0 * M * N * K) / (ms * 1.0e6);
	std::printf("MatMul: %dx%dx%d, time=%.3f ms, GFLOPS=%.2f, max_err=%e\n",
							M, N, K, ms, gflops, max_err);

	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	return 0;
}
