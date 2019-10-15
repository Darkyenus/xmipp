/***************************************************************************
 *
 * Authors:     Jan Polak (456647@mail.muni.cz)
 *              (some code derived from other Xmipp programs by other authors)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#include <core/multidim_array.h>

#include <reconstruction_cuda/cuda_xmipp_utils.h>
#include <reconstruction_cuda/cuda_asserts.h>

#include <math.h>
#include <fftw3.h>
#include <starpu.h>
#include <pthread.h>

#include <core/xmipp_fft.h>
#include <core/xmipp_fftw.h>

#include "reconstruct_fourier_codelets.h"
#include "reconstruct_fourier_defines.h"
#include "reconstruct_fourier_util.h"

/*
 * Codelet will create and process the 'paddedFourier' (not shifted, i.e. low frequencies are in corners) in the following way:
 * - high frequencies are skipped (replaced by zero (0))
 * - space is shifted, so that low frequencies are in the middle of the Y axis
 * - resulting space is cropped
 * - values are normalized
 * Codelet outputs a 2D array with Fourier coefficients, shifted so that low frequencies are
 * in the center of the Y axis (i.e. semicircle).
 */

/**
 * Index to frequency. Same as xmipp_fft.h::FFT_IDX2DIGFREQ, but compatible with CUDA and not using doubles.
 * Given an index and a size of the FFT, this function returns the corresponding digital frequency (-1/2 to 1/2).
 */
__host__ __device__
inline float fft_IDX2DIGFREQ(int idx, int size) {
	if (size <= 1) return 0;
	return ((idx <= (size / 2)) ? idx : (-size + idx)) / (float) size;
}

// ======================================= CPU =====================================

static void cropAndShift(
		const float2* input,
		uint32_t inputSizeX,
		uint32_t inputSizeY,
		uint32_t outputSizeX,
		float2* output,
		const float maxResolutionSqr,
		const float normalizationFactor) {

	// convert image (shift to center and remove high frequencies)
	int halfY = static_cast<int>(inputSizeY / 2);
	for (uint32_t y = 0; y < inputSizeY; y++) {
		for (uint32_t x = 0; x < outputSizeX; x++) {
			if (y < outputSizeX || y >= (inputSizeY - outputSizeX)) {
				// check the frequency
				float2 freq = { fft_IDX2DIGFREQ(x, inputSizeY),
				                fft_IDX2DIGFREQ(y, inputSizeY) };

				float2 item;
				if (freq.x * freq.x + freq.y * freq.y > maxResolutionSqr) {
					item = float2 { 0.0f, 0.0f };
				} else {
					item = input[y * inputSizeX + x];
					item.x *= normalizationFactor;
					item.y *= normalizationFactor;
				}
				// do the shift
				int myPadI = static_cast<int>((y < halfY) ? y + outputSizeX : y - inputSizeY + outputSizeX);
				int index = static_cast<int>(myPadI * outputSizeX + x);
				output[index] = item;
			}
		}
	}
}

static void frequencyDomainShiftCpu(float2* image, uint32_t sizeX, uint32_t sizeY, uint32_t memorySizeX, float shiftX, float shiftY) {
	// https://www.arc.id.au/ZoomFFT.html
	// https://www.clear.rice.edu/elec301/Projects01/image_filt/properties.html#shiftp
	// https://stackoverflow.com/questions/25827916/matlab-shifting-an-image-using-fft
	// http://www.thefouriertransform.com/transform/properties.php
	const float factorX = shiftX / sizeX;
	const float factorY = shiftY / sizeY;
	for (uint32_t y = 0; y < sizeY; ++y) {
		float2* imageRow = image + y * memorySizeX;
		for (uint32_t x = 0; x < memorySizeX; ++x) {
			float2* imagePixel = imageRow + x;

			const float oldReal = imagePixel->x;
			//float newReal = oldReal * cosf(TWOPI * (shiftX * x / sizeX + shiftY * y / sizeY));
			//float newImaginary = oldReal * -sinf(TWOPI * (shiftX * x / sizeX + shiftY * y / sizeY));
			const float angle = TWOPI * (factorX * x + factorY * y);
			float newReal = oldReal * cosf(angle);
			float newImaginary = oldReal * -sinf(angle);

			imagePixel->x = newReal;
			imagePixel->y = newImaginary;
		}
	}
}

static void testFrequencyDomainShift() {
	static bool tested = false;
	if (tested) {
		return;
	}
	tested = true;


	/*
	 int size = 4;
	 float2 image[] = {
			{1, + 0},  {0, - 1},{-1, + 0},  {0, + 1},
			{0, - 1} , {-1, + 0}  , {0, + 1}  ,  {1, + 0},
			{-1, + 0},   {0, + 1} ,  {1, + 0} ,   {0, - 1},
			{0, + 1} ,  {1, - 0}  , {0, - 1}  , {-1, - 0}
	};*/
	/*
1 0i   0 0i    1 0   0 0i
0 0i     0 1i   0 0i   0 1i
-1 0i   0 0i   -1 0i   0 0i
0 0i   0 -1i   0 0i   0 -1i
	 */

	/*int size = 2;
	float2 image[] = {
			{1, 0}, {-1, 0},
			{1, 0}, {-1, 0}
	};*/

	int size = 3;
	float2 image[] = {
{1.00000, + 0.00000i }, { -0.50000, - 0.86603i}, {-0.50000, + 0.86603i},
{-0.50000, - 0.86603i}, { -0.50000, + 0.86603i}, { 1.00000, + 0.00000i},
{-0.50000, + 0.86603i}, {  1.00000, - 0.00000i}, {-0.50000, - 0.86603i},
	};
	/*
	 Expected:
	 1.00000 + 0.00000i  -0.50000 + 0.86603i  -0.50000 - 0.86603i
	-0.50000 - 0.86603i   1.00000 + 0.00000i  -0.50000 + 0.86603i
	-0.50000 + 0.86603i  -0.50000 - 0.86603i   1.00000 - 0.00000i

	 Got:

	 */

	frequencyDomainShiftCpu(image, size, size, size, 1, 0);
	/*
	Expected:
	1   1
    1   1

	 Got:
	 1 -0i   1 -8.74228e-08i
	 1 -0i   1 -8.74228e-08i
	 */

	for (int y = 0; y < size; ++y) {
		for (int x = 0; x < size; ++x) {
			float2 p = image[y * size + x];
			std::cout << p.x << " " << p.y << "i   ";
		}
		std::cout << "\n";
	}
}

void func_padded_image_to_fft_cpu(void **buffers, void *cl_arg) {
	testFrequencyDomainShift();
	float* inPaddedImage = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
	float2* outProcessedFft = (float2*)STARPU_VECTOR_GET_PTR(buffers[1]);
	float2* temporaryFftScratch = (float2*)STARPU_MATRIX_GET_PTR(buffers[2]);
	const uint32_t noOfImages = ((LoadedImagesBuffer*) STARPU_VARIABLE_GET_PTR(buffers[3]))->noOfImages;
	const auto* transformArgs = (FrequencyDomainTransformArgs*)STARPU_VECTOR_GET_PTR(buffers[4]);
	const PaddedImageToFftArgs& arg = *(PaddedImageToFftArgs*)(cl_arg);

	const uint32_t rawFftSizeX = STARPU_MATRIX_GET_NX(buffers[2]);
	const uint32_t rawFftSizeY = STARPU_MATRIX_GET_NY(buffers[2]);

	const size_t imageStridePaddedImage = STARPU_VECTOR_GET_ELEMSIZE(buffers[0]) / sizeof(float);
	const size_t temporaryFftScratchSizeBytes = STARPU_MATRIX_GET_ELEMSIZE(buffers[2]) * rawFftSizeX * rawFftSizeY;
	const size_t imageStrideOutput = STARPU_VECTOR_GET_ELEMSIZE(buffers[1]) / sizeof(float2);

	if (alignmentOf(inPaddedImage) < ALIGNMENT || alignmentOf(temporaryFftScratch) < ALIGNMENT) {
		fprintf(stderr,"Bad alignments of buffers for FFT: %d, %d\n", alignmentOf(inPaddedImage), alignmentOf(temporaryFftScratch));
		assert(false);
	}

	static pthread_mutex_t fftw_plan_mutex = PTHREAD_MUTEX_INITIALIZER;
	// Planning API of FFTW is not thread safe
	pthread_mutex_lock(&fftw_plan_mutex);
	fftwf_plan plan = fftwf_plan_dft_r2c_2d(arg.paddedImageSize, arg.paddedImageSize,
			inPaddedImage, (fftwf_complex*) temporaryFftScratch, FFTW_ESTIMATE);
	pthread_mutex_unlock(&fftw_plan_mutex);

	assert(plan != nullptr);

	const float normalizationFactor = 1.0f / (arg.paddedImageSize * arg.paddedImageSize);

	float* in = inPaddedImage;
	float2* out = outProcessedFft;
	for (uint32_t i = 0; i < noOfImages; i++) {
		memset(temporaryFftScratch, 0, temporaryFftScratchSizeBytes);

		fftwf_execute_dft_r2c(plan, in, (fftwf_complex*) temporaryFftScratch);

		frequencyDomainShiftCpu(temporaryFftScratch, arg.paddedImageSize, arg.paddedImageSize, rawFftSizeX, transformArgs[i].shiftX, transformArgs[i].shiftY);

		cropAndShift(temporaryFftScratch, rawFftSizeX, rawFftSizeY, arg.fftSizeX,
		             out, arg.maxResolutionSqr, normalizationFactor);

		in += imageStridePaddedImage;
		out += imageStrideOutput;
	}

	pthread_mutex_lock(&fftw_plan_mutex);
	fftwf_destroy_plan(plan);
	pthread_mutex_unlock(&fftw_plan_mutex);
}

// ======================================= CUDA =====================================

__global__
void convertImagesKernel(
		const float2* input,
		uint32_t inputSizeX,
		uint32_t inputSizeY,
		uint32_t outputSizeX,//fftSizeX
		float2* output,
		const float maxResolutionSqr,
		const float normFactor) {
	// assign pixel to thread
	volatile int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	volatile int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

	int halfY = inputSizeY / 2;

	// input is an image in Fourier space (not normalized)
	// with low frequencies in the inner corners
	float2 freq;
	if ((idy < inputSizeY) // for all input lines
	    && (idx < outputSizeX)) { // for all output pixels in the line
		// process line only if it can hold sufficiently high frequency, i.e. process only
		// first and last N lines
		if (idy < outputSizeX || idy >= (inputSizeY - outputSizeX)) {
			// check the frequency
			freq.x = fft_IDX2DIGFREQ(idx, inputSizeY);
			freq.y = fft_IDX2DIGFREQ(idy, inputSizeY);
			if ((freq.x * freq.x + freq.y * freq.y) > maxResolutionSqr) {
				return;
			}
			// do the shift (lower line will move up, upper down)
			int newY = (idy < halfY) ? (idy + outputSizeX) : (idy - inputSizeY + outputSizeX);
			int oIndex = newY * outputSizeX + idx;

			// copy data and perform normalization
			int iIndex = idy*inputSizeX + idx;
			float2 value = input[iIndex];
			value.x *= normFactor;
			value.y *= normFactor;
			output[oIndex] = value;
		}
	}
}

__global__
static void frequencyDomainShiftGpu(float2* image, uint32_t memorySizeX, uint32_t memorySizeY, float factorX, float factorY) {
	// CUDA version of frequencyDomainShiftCpu
	uint32_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= memorySizeX || y >= memorySizeY) {
		return;
	}

	float2* imagePixel = image + y * memorySizeX + x;

	const float oldReal = imagePixel->x;
	const float angle = TWOPI * (factorX * x + factorY * y);
	const float newReal = oldReal * cosf(angle);
	const float newImaginary = oldReal * -sinf(angle);

	imagePixel->x = newReal;
	imagePixel->y = newImaginary;
}

void func_padded_image_to_fft_cuda(void **buffers, void *cl_arg) {
	testFrequencyDomainShift();
	float* inPaddedImage = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
	float2* outProcessedFft = (float2*)STARPU_VECTOR_GET_PTR(buffers[1]);
	float2* temporaryFftScratch = (float2*)STARPU_MATRIX_GET_PTR(buffers[2]);
	const uint32_t noOfImages = ((LoadedImagesBuffer*) STARPU_VARIABLE_GET_PTR(buffers[3]))->noOfImages;
	const auto* transformArgs = (FrequencyDomainTransformArgs*)STARPU_VECTOR_GET_PTR(buffers[4]);
	const PaddedImageToFftArgs& arg = *(PaddedImageToFftArgs*)(cl_arg);

	const uint32_t rawFftSizeX = STARPU_MATRIX_GET_NX(buffers[2]);
	const uint32_t rawFftSizeY = STARPU_MATRIX_GET_NY(buffers[2]);

	const size_t imageStridePaddedImage = STARPU_VECTOR_GET_ELEMSIZE(buffers[0]) / sizeof(float);
	const size_t temporaryFftScratchSizeBytes = STARPU_MATRIX_GET_ELEMSIZE(buffers[2]) * rawFftSizeX * rawFftSizeY;
	const size_t imageStrideOutput = STARPU_VECTOR_GET_ELEMSIZE(buffers[1]) / sizeof(float2);

	if (alignmentOf(inPaddedImage) < ALIGNMENT || alignmentOf(temporaryFftScratch) < ALIGNMENT) {
		fprintf(stderr,"Bad alignments of buffers for FFT: %d, %d\n", alignmentOf(inPaddedImage), alignmentOf(temporaryFftScratch));
		assert(false);
	}

	//TODO Investigate the use of cuFFTAdvisor to achieve better performance
	//TODO Cache the plan explicitly
	cufftHandle plan;
	gpuErrchkFFT(cufftPlan2d(&plan, arg.paddedImageSize, arg.paddedImageSize, cufftType::CUFFT_R2C));
	gpuErrchkFFT(cufftSetStream(plan, starpu_cuda_get_local_stream()));

	const float normalizationFactor = 1.0f / (arg.paddedImageSize * arg.paddedImageSize);

	float* in = inPaddedImage;
	float2* out = outProcessedFft;
	for (uint32_t i = 0; i < noOfImages; i++) {
		// Clear the memory to which we will write
		// Clear FFT output
		gpuErrchk(cudaMemsetAsync(temporaryFftScratch, 0, temporaryFftScratchSizeBytes, starpu_cuda_get_local_stream()));
		// Clear cropAndShift output (as the kernel writes only somewhere)
		//TODO It could be cheaper to just write everywhere...
		gpuErrchk(cudaMemsetAsync(out, 0, imageStrideOutput * sizeof(float2), starpu_cuda_get_local_stream()));

		// Execute FFT plan
		gpuErrchkFFT(cufftExecR2C(plan, in, temporaryFftScratch));

		// One thread for each pixel of raw FFT
		dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
		dim3 dimGrid((rawFftSizeX + dimBlock.x - 1) / dimBlock.x, (rawFftSizeY + dimBlock.y - 1) / dimBlock.y);

		// Translate FFT
		frequencyDomainShiftGpu<<<dimGrid, dimBlock, 0, starpu_cuda_get_local_stream()>>>(
				temporaryFftScratch, rawFftSizeX, rawFftSizeY,
				transformArgs[i].shiftX / arg.paddedImageSize, transformArgs[i].shiftY / arg.paddedImageSize);
		gpuErrchk(cudaPeekAtLastError());

		// Process results
		convertImagesKernel<<<dimGrid, dimBlock, 0, starpu_cuda_get_local_stream()>>>(
				temporaryFftScratch, rawFftSizeX, rawFftSizeY, arg.fftSizeX,
				out, arg.maxResolutionSqr, normalizationFactor);
		gpuErrchk(cudaPeekAtLastError());

		in += imageStridePaddedImage;
		out += imageStrideOutput;
	}

	cufftDestroy(plan);

	// gpuErrchk(cudaStreamSynchronize(starpu_cuda_get_local_stream())); disabled because codelet is async
}
