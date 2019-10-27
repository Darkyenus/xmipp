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

/**
 * For fft shift index i returns a value gained by first subtracting size/2 from it and then CenterFFT/fftshift shifting it.
 * For example for indices 0 to 12 and size 13 returns:
 * 0   1   2   3   4   5   6  -6  -5  -4  -3  -2  -1
 */
inline int fftIndexShift(int i, int size) {
	int halfSize = size / 2;
	if (i <= halfSize) {
		return i;
	} else {
		return i - size;
	}
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
	// https://stackoverflow.com/questions/25827916/matlab-shifting-an-image-using-fft
	// https://www.clear.rice.edu/elec301/Projects01/image_filt/properties.html#shiftp
	// http://www.thefouriertransform.com/transform/properties.php
	const float factorX = shiftX / sizeX;
	const float factorY = shiftY / sizeY;
	for (uint32_t y = 0; y < sizeY; ++y) {
		float2* imageRow = image + y * memorySizeX;
		for (uint32_t x = 0; x < memorySizeX; ++x) {
			const float angle = -((float)TWOPI) * ((factorX * fftIndexShift(x, sizeX)) + (factorY * fftIndexShift(y, sizeY)));
			const float factorReal = cosf(angle);
			const float factorImag = sinf(angle);

			float2* imagePixel = imageRow + x;
			const float pixelReal = imagePixel->x;
			const float pixelImag = imagePixel->y;
			const float newReal = pixelReal * factorReal - pixelImag * factorImag;
			const float newImag = pixelReal * factorImag + pixelImag * factorReal;
			imagePixel->x = newReal;
			imagePixel->y = newImag;
		}
	}
}

static void frequencyDomainShiftCpu(double2* image, uint32_t sizeX, uint32_t sizeY, uint32_t memorySizeX, double shiftX, double shiftY) {
	// https://www.arc.id.au/ZoomFFT.html
	// https://stackoverflow.com/questions/25827916/matlab-shifting-an-image-using-fft
	// https://www.clear.rice.edu/elec301/Projects01/image_filt/properties.html#shiftp
	// http://www.thefouriertransform.com/transform/properties.php
	const double factorX = shiftX / sizeX;
	const double factorY = shiftY / sizeY;
	for (uint32_t y = 0; y < sizeY; ++y) {
		double2* imageRow = image + y * memorySizeX;
		for (uint32_t x = 0; x < memorySizeX; ++x) {
			const double angle = -(TWOPI) * ((factorX * fftIndexShift(x, sizeX)) + (factorY * fftIndexShift(y, sizeY)));
			const double factorReal = cos(angle);
			const double factorImag = sin(angle);

			double2* imagePixel = imageRow + x;
			const double pixelReal = imagePixel->x;
			const double pixelImag = imagePixel->y;
			const double newReal = pixelReal * factorReal - pixelImag * factorImag;
			const double newImag = pixelReal * factorImag + pixelImag * factorReal;
			imagePixel->x = newReal;
			imagePixel->y = newImag;
		}
	}
}


#include <data/projection.h>
#include <math.h>
#include <core/xmipp_fft.h>
#include <reconstruction_cuda/cuda_gpu_geo_shift_transformer.cpp>

template<typename T>
static void generateTestImage(Image<T>& testImage, int testImageSize) {
	auto& testImageData = testImage.data;
	testImageData.resizeNoCopy(testImageSize, testImageSize);
	testImageData.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY2D(testImageData) {
			if (j < 0) {
				if (j*j + i*i < (testImageSize/4) * (testImageSize/4)) {
					A2D_ELEM(testImageData, i, j) = 1;
				} else {
					A2D_ELEM(testImageData, i, j) = 0;
				}
			} else {
				if (abs(j) + abs(i) < (testImageSize/4)) {
					A2D_ELEM(testImageData, i, j) = 1;
				} else {
					A2D_ELEM(testImageData, i, j) = 0;
				}
			}
		}
}

static void testFrequencyDomainShift() {
	static bool tested = false;
	if (tested) {
		return;
	}
	tested = true;

	int testImageSize = 5;
	double shiftX = 15.5;
	double shiftY = -13;

	{
		Image<double> testImage;
		generateTestImage(testImage, testImageSize);
		testImage.write(FileName("SHIFT_TEST_before.tiff"));
		testImage.setShifts(shiftX, shiftY);
		testImage.setFlip(false);
		testImage.selfApplyGeometry(3, true, true);
		testImage.write(FileName("SHIFT_TEST_normal.tiff"));
	}

	{
		Image<double> testImage;
		generateTestImage(testImage, testImageSize);

		MultidimArray<std::complex<double>> testImageFFT;
		FourierTransform(testImage.data, testImageFFT);

		float2* testImageFFTFloat = (float2*) malloc(sizeof(float2) * testImageFFT.getSize());
		std::cout << std::fixed << std::setprecision(3) << "after_fft = [ ";
		for (int i = 0; i < testImageFFT.getSize(); ++i) {
			std::complex<double>& c = testImageFFT.data[i];
			testImageFFTFloat[i] = { c.real(), c.imag() };
			std::cout << "(" << c.real() << "+" << c.imag() << +"i) ";
			if ((i + 1) % testImageFFT.xdim == 0 && i + 1 < testImageFFT.getSize()) {
				std::cout << ";\n";
			}
		}
		std::cout<<"]\n";

		frequencyDomainShiftCpu(testImageFFTFloat,
				(uint32_t)testImageFFT.getDimensions().xdim,
				(uint32_t)testImageFFT.getDimensions().ydim,
				(uint32_t)testImageFFT.getDimensions().xdim, (float)shiftX, (float)shiftY);

		std::cout << std::fixed << std::setprecision(8) << "after_shift = [ ";
		for (int i = 0; i < testImageFFT.getSize(); ++i) {
			float2& c = testImageFFTFloat[i];
			testImageFFT.data[i] = std::complex<double>(c.x, c.y);
			std::cout << "(" << c.x << "+" << c.y << +"i) ";
			if ((i + 1) % testImageFFT.xdim == 0 && i + 1 < testImageFFT.getSize()) {
				std::cout << ";\n";
			}
		}
		std::cout<<"]\n";


		free(testImageFFTFloat);
		InverseFourierTransform(testImageFFT, testImage.data);

		testImage.write(FileName("SHIFT_TEST_fft.tiff"));
	}

	{
		Image<double> testImage;
		generateTestImage(testImage, testImageSize);

		MultidimArray<std::complex<double>> testImageFFT;
		FourierTransform(testImage.data, testImageFFT);

		frequencyDomainShiftCpu((double2*)testImageFFT.data,
		                        (uint32_t)testImageFFT.getDimensions().xdim,
		                        (uint32_t)testImageFFT.getDimensions().ydim,
		                        (uint32_t)testImageFFT.getDimensions().xdim, shiftX, shiftY);

		InverseFourierTransform(testImageFFT, testImage.data);

		testImage.write(FileName("SHIFT_TEST_fft_double.tiff"));
	}

	{
		Image<float> testImage;
		generateTestImage(testImage, testImageSize);

		GeoShiftTransformer<float> transformer;
		transformer.init(GPU(), testImageSize, testImageSize, 1, 0, nullptr);

		Image<float> testImageOutput;
		transformer.applyShift(testImageOutput.data, testImage.data, shiftX, shiftY);

		testImageOutput.write(FileName("SHIFT_TEST_fft_strelak.tiff"));
	}

	/*
	int size = 3;
	float2 image[] = {
		{1.03268 , + 0.00000}, { -0.57095, - 0.86842},{ -0.57095, + 0.86842},
		{-0.88368, - 0.58689}, { -0.01675, + 1.09332},{  0.95534, - 0.41150},
		{-0.88368, + 0.58689}, {  0.95534, + 0.41150},{ -0.01675, - 1.09332}
	};

	frequencyDomainShiftCpu(image, size, size, size, 1, 0);

	for (int y = 0; y < size; ++y) {
		for (int x = 0; x < size; ++x) {
			float2 p = image[y * size + x];
			std::cout << std::fixed << std::setprecision(3) << p.x << " " << p.y << "i   ";
		}
		std::cout << "\n";
	}
	 */
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
