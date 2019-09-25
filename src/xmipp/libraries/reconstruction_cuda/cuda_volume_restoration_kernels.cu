#include "cuda_cdf.cu"

#include "cuda_basic_math.h"

namespace Gpu {

template< typename T >
__global__ void computeWeightsKernel(const T* d_Vfiltered1, const T* d_Vfiltered2, T* d_V1r, T* d_V2r, T* d_S,
			const T* d_x, const T* d_probXLessThanx, const T* d_V, size_t volume_size, size_t Nsteps,
			T weightPower, int weightFun) {

	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n >= volume_size) {
		return;
	}

	T minVal = d_V[0];
	T maxVal = d_V[volume_size - 1];

	T f1 = d_Vfiltered1[n];
	T e1 = f1 * f1;
	T w1 = Gpu::getCDFProbability(e1, d_x, d_probXLessThanx, Nsteps, minVal, maxVal);

	T f2 = d_Vfiltered2[n];
	T e2 = f2 * f2;
	T w2 = Gpu::getCDFProbability(e2, d_x, d_probXLessThanx, Nsteps, minVal, maxVal);

    T weight;
    switch (weightFun) {
		case 0: weight = 0.5 * (w1 + w2); break;
		case 1: weight = min(w1, w2); break;
		case 2: weight = 0.5 * (w1 + w2) * (1 - abs(w1 - w2)/(w1 + w2)); break;
    }
    weight = power(weight, weightPower);

    T Vf1w = f1 * weight;
    T Vf2w = f2 * weight;
    d_V1r[n] += Vf1w;
    d_V2r[n] += Vf2w;
    if (e1 > e2) {
    	d_S[n] += Vf1w;
    } else {
    	d_S[n] += Vf2w;
    }
}

template< typename T >
__global__ void filterFourierVolumeKernel(const T* d_R2, const vec2_type<T>* d_fV, vec2_type<T>* d_buffer, size_t volume_size, T w2, T w2Step) {

	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n >= volume_size) {
		return;
	}

	T R2n = d_R2[n];
	if (R2n >= w2 && R2n < w2Step) {
		d_buffer[n] = d_fV[n];
	} else {
		d_buffer[n].x = 0;
		d_buffer[n].y = 0;
	}
}

template< typename T >
__global__ void computeAveragePositivityKernel(const T* d_V1, const T* d_V2, T* d_S, size_t volume_size, T inv_size) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n >= volume_size) {
		return;
	}

	T val = 0.5 * (d_V1[n] + d_V2[n]);
	d_S[n] = val;

	if (val <= 0) {
		d_S[n] = 0;
	}
}

template< typename T >
__global__ void computeAveragePositivityKernel(const T* d_V1, const T* d_V2, T* d_S, const int* d_mask, size_t volume_size, T inv_size) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n >= volume_size) {
		return;
	}

	T val = 0.5 * (d_V1[n] + d_V2[n]);
	d_S[n] = val;

	if (val <= 0 || d_mask[n] == 0) {
		d_S[n] = 0;
	}
}

template< typename T >
__global__ void filterSKernel(const T* d_R2, vec2_type<T>* d_fVol, size_t volume_size) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n >= volume_size) {
		return;
	}

	if (d_R2[n] > static_cast<T>(0.25)) {
		d_fVol[n].x = 0.0;
		d_fVol[n].y = 0.0;
	}
}

template< typename T >
__global__ void maskWithNoiseProbabilityKernel(T* d_V, const T* d_xS, const T* d_probXLessThanxS, const T* d_VS, size_t NstepsS, size_t Ssize,
										const T* d_xN, const T* d_probXLessThanxN, const T* d_VN, size_t NstepsN, size_t Nsize) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n >= Nsize) {
		return;
	}

	T e = d_V[n] * d_V[n];
	T pN = Gpu::getCDFProbability(e, d_xN, d_probXLessThanxN, NstepsN, d_VN[0], d_VN[Nsize - 1]);

	if (pN < 1) {
		pN *= Gpu::getCDFProbability(e, d_xS, d_probXLessThanxS, NstepsS, d_VS[0], d_VS[Ssize - 1]);
		d_V[n] = pN * d_V[n];
	}
}

template< typename T >
__global__ void deconvolveRestoredKernel(vec2_type<T>* d_fVol, vec2_type<T>* d_fV1, vec2_type<T>* d_fV2, const T* d_R2, T K1, T K2, T lambda, size_t volume_size, T inv_size) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n >= volume_size) {
		return;
	}

	T R2n = d_R2[n];

	if (R2n <= static_cast<T>(0.25)) {
		T H1 = exp(K1 * R2n);
		T H2 = exp(K2 * R2n);

		d_fVol[n].x = (H1*d_fV1[n].x + H2*d_fV2[n].x) / (H1*H1 + H2*H2 + lambda*R2n);
		d_fVol[n].y = (H1*d_fV1[n].y + H2*d_fV2[n].y) / (H1*H1 + H2*H2 + lambda*R2n);

		H1 = static_cast<T>(1.0) / H1;
		H2 = static_cast<T>(1.0) / H2;

		d_fV1[n].x *= H1;
		d_fV1[n].y *= H1;
		d_fV2[n].x  *= H2;
		d_fV2[n].y  *= H2;

	}
}

template< typename T >
__global__ void convolveFourierVolumeKernel(vec2_type<T>* d_fVol, const T* d_R2, T K, size_t volume_size) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n >= volume_size) {
		return;
	}

	T R2n = d_R2[n];
	if (R2n <= static_cast<T>(0.25)) {
		d_fVol[n].x *= exp(K * R2n);
		d_fVol[n].y *= exp(K * R2n);
	}
}

template< typename T >
__global__ void normalizeForFFTKernel(T* d_V1, T* d_V2, size_t volume_size, T inv_size) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n >= volume_size) {
		return;
	}

	d_V1[n] *= inv_size;
	d_V2[n] *= inv_size;
}

template< typename T >
__global__ void normalizeForFFTKernel(T* d_V1, size_t volume_size, T inv_size) {
	int n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n >= volume_size) {
		return;
	}

	d_V1[n] *= inv_size;
}

} // namespace Gpu