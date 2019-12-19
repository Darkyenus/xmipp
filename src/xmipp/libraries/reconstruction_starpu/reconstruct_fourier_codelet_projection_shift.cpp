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

#include "reconstruct_fourier_codelets.h"
#include "core/transformations.h"
#include "core/xmipp_fft.h"
#include <cstring>
#include <cassert>
#include <memory>

//               ================ CPU ==================

void applyTransformation(const float* __restrict__ source, float* __restrict__ target,
                   const int32_t sizeX, const int32_t sizeY, const Matrix2D<double> &mat) {
	// 2D transformation
	double Aref00=MAT_ELEM(mat, 0, 0);
	double Aref10=MAT_ELEM(mat, 1, 0);

	// Find center and limits of image
	const int32_t cen_y  = sizeY / 2;
	const int32_t cen_x  = sizeX / 2;
	const int32_t minxp  = -cen_x;
	const int32_t minyp  = -cen_y;
	const double minxpp = minxp-XMIPP_EQUAL_ACCURACY;
	const double minypp = minyp-XMIPP_EQUAL_ACCURACY;
	const int32_t maxxp  = sizeX - cen_x - 1;
	const int32_t maxyp  = sizeY - cen_y - 1;
	const double maxxpp = maxxp+XMIPP_EQUAL_ACCURACY;
	const double maxypp = maxyp+XMIPP_EQUAL_ACCURACY;

	// Since following code requires doubles, we have to copy+cast
	std::unique_ptr<double[]> sourceDouble(new double[sizeX * sizeY]);
	for (int32_t c = 0; c < sizeX * sizeY; ++c) {
		sourceDouble[c] = (double) source[c];
	}

	// Build the B-spline coefficients
	MultidimArray<double> Bcoeffs;
	Bcoeffs.initZeros(sizeY, sizeX);

	int Status;
	ChangeBasisVolume(sourceDouble.get(), MULTIDIM_ARRAY(Bcoeffs),
	                  sizeX, sizeY, 1,
	                  CardinalSpline, BasicSpline, 3,
	                  MirrorOffBounds, DBL_EPSILON, &Status);
	if (Status != 0) {
		REPORT_ERROR(ERR_UNCLASSIFIED, "Error in produceSplineCoefficients...");
	}

	STARTINGX(Bcoeffs) = (int) minxp;
	STARTINGY(Bcoeffs) = (int) minyp;

	// Now we go from the output image to the input image, ie, for any pixel
	// in the output image we calculate which are the corresponding ones in
	// the original image, make an interpolation with them and put this value
	// at the output pixel

	// Calculate position of the beginning of the row in the output image
	double sourceX = -cen_x;
	double sourceY = -cen_y;
	for (int32_t targetY = 0; targetY < sizeY; targetY++) {
		// Calculate this position in the input image according to the
		// geometrical transformation they are related by
		// coords_output(=x,y) = A * coords_input (=xp,yp)
		double xp = sourceX * MAT_ELEM(mat, 0, 0) + sourceY * MAT_ELEM(mat, 0, 1) + MAT_ELEM(mat, 0, 2);
		double yp = sourceX * MAT_ELEM(mat, 1, 0) + sourceY * MAT_ELEM(mat, 1, 1) + MAT_ELEM(mat, 1, 2);

		// This is original implementation
		for (int32_t targetX = 0; targetX < sizeX; targetX++) {
			if (XMIPP_RANGE_OUTSIDE_FAST(xp, minxpp, maxxpp)) {
				xp = realWRAP(xp, minxp - 0.5, maxxp + 0.5);
			}

			if (XMIPP_RANGE_OUTSIDE_FAST(yp, minypp, maxypp)) {
				yp = realWRAP(yp, minyp - 0.5, maxyp + 0.5);
			}

			// B-spline interpolation
			target[targetX + targetY * sizeX] = (float) Bcoeffs.interpolatedElementBSpline2D_Degree3(xp, yp);

			// Compute new point inside input image
			xp += Aref00;
			yp += Aref10;
		}

		sourceY++;
	}
}

void func_projection_shift_cpu(void **buffers, void *cl_arg) {
	const size_t projectionStride = STARPU_VECTOR_GET_ELEMSIZE(buffers[0]) / sizeof(float);
	const auto* inProjection = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
	auto* outProjection = (float*)STARPU_VECTOR_GET_PTR(buffers[1]);

	const ProjectionShiftArgs& arg = *((ProjectionShiftArgs*) cl_arg);

	const auto* transformArgs = (ProjectionShiftBuffer*)STARPU_VECTOR_GET_PTR(buffers[2]);
	const uint32_t noOfImages = ((LoadedImagesBuffer*) STARPU_VARIABLE_GET_PTR(buffers[3]))->noOfImages;

	// Following code replaces original call to:
	// applyGeometry(BSPLINE3, transformedImageData, proj.data, shiftMatrix, IS_NOT_INV, WRAP);


	// Reset output to zeros, so we don't have to do it later
	memset(outProjection, 0, noOfImages * projectionStride);

	Matrix2D<double> shiftMatrix;
	Matrix2D<double> shiftMatrixInv;
	for (uint32_t i = 0; i < noOfImages; ++i) {
		shiftMatrix.initIdentity(3);
		const ProjectionShiftBuffer& shiftArg = transformArgs[i];
		dMij(shiftMatrix, 0, 2) = shiftArg.shiftX;
		dMij(shiftMatrix, 1, 2) = shiftArg.shiftY;
		/*if (shiftArg.flip) {
			dMij(shiftMatrix, 0, 0) *= -1.;
			dMij(shiftMatrix, 0, 1) *= -1.;
		}*/
		shiftMatrix.inv(shiftMatrixInv);

		applyTransformation(inProjection + i * projectionStride, outProjection + i * projectionStride,
				(int32_t)arg.projectionSize, (int32_t)arg.projectionSize, shiftMatrix);

		// CenterFFT = center for fft (not fft itself)
		// NOTE(jp): I am not sure why is this done. It seems to flip some signs in the result of FFT according to some pattern.
		// TODO(jp): Could this be replaced with a shift by sizeX/2 and sizeY/2?

		MultidimArray<float> viewForCentering;
		viewForCentering.data = outProjection + i * projectionStride;
		viewForCentering.destroyData = false;
		viewForCentering.setDimensions(arg.projectionSize, arg.projectionSize, 1, 1);
		CenterFFT(viewForCentering, true); //TODO(jp): This uses further internal copying
	}
}

//               ================ CUDA ==================

void func_projection_shift_cuda(void **buffers, void *cl_arg) {
	const auto* inProjection = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
	auto* outProjection = (float*)STARPU_VECTOR_GET_PTR(buffers[1]);
	const auto* transformArgs = (ProjectionShiftBuffer*)STARPU_VECTOR_GET_PTR(buffers[2]);
	const uint32_t noOfImages = ((LoadedImagesBuffer*) STARPU_VARIABLE_GET_PTR(buffers[3]))->noOfImages;

	assert(false);
}