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

#include <iostream>

#include <core/args.h>
#include <core/metadata.h>
#include <data/projection.h>
#include <core/xmipp_fftw.h>

#include "reconstruct_fourier_codelets.h"
#include "reconstruct_fourier_starpu_util.h"

void func_load_projections(void* buffers[], void* cl_arg) {
	const LoadProjectionArgs& arg = *static_cast<LoadProjectionArgs*>(cl_arg);

	LoadProjectionAmountLoaded& amountLoaded = *((LoadProjectionAmountLoaded*)STARPU_VARIABLE_GET_PTR(buffers[0]));
	float* outImageData = (float*)STARPU_VECTOR_GET_PTR(buffers[1]);
	const size_t outImageDataStride = STARPU_VECTOR_GET_ELEMSIZE(buffers[1]) / sizeof(float);
	RecFourierProjectionTraverseSpace* outSpaces = (RecFourierProjectionTraverseSpace*)STARPU_VECTOR_GET_PTR(buffers[2]);

	ApplyGeoParams geoParams;
	geoParams.only_apply_shifts = true;

	MultidimArray<float> paddedImageData; // Declared here so that internal allocated memory can be reused

	uint32_t traverseSpaceIndex = 0;
	uint32_t projectionIndex = 0;

	for (uint32_t projectionInBatch = arg.batchStart; projectionInBatch < arg.batchEnd; projectionInBatch++) {
		const size_t imageObjectIndex = arg.imageObjectIndices[projectionInBatch];

		//Read projection from selfile, read also angles and shifts if present
		//but only apply shifts (set above)
		// FIXME following line is a current bottleneck, as it calls BSpline interpolation
		Projection proj;
		proj.readApplyGeo(arg.selFile, imageObjectIndex, geoParams);
		if (arg.useWeights && proj.weight() == 0.f) {
			continue;
		}
		// NOTE(jp): No `continue` skipping after this point, indices to outputs are in lockstep

		// Compute the coordinate axes associated to this projection
		Matrix2D<double> Ainv(3, 3);
		Euler_angles2matrix(proj.rot(), proj.tilt(), proj.psi(), Ainv);
		Ainv = Ainv.transpose();

		// prepare transforms for all  symmetries
		for (const Matrix2D<double>& symmetry : arg.rSymmetries) {
			RecFourierProjectionTraverseSpace& space = outSpaces[traverseSpaceIndex++];
			space.weight = arg.useWeights ? static_cast<float>(proj.weight()) : 1.0f;
			space.projectionIndex = projectionIndex; // "index to some array where the respective projection is stored"

			Matrix2D<double> A_SL = symmetry * Ainv;
			Matrix2D<double> A_SLInv = A_SL.inv();
			float transf[3][3];
			float transfInv[3][3];
			A_SL.convertTo(transf);
			A_SLInv.convertTo(transfInv);

			computeTraverseSpace(arg.fftSizeX, arg.fftSizeY,
			                     transf, transfInv, space,
			                     arg.maxVolumeIndexX, arg.maxVolumeIndexYZ, arg.fastLateBlobbing, arg.blobRadius);

		}

		// Copy the projection to the center of the padded image
		// and compute its Fourier transform, if requested
		proj().setXmippOrigin();
		const MultidimArray<double> &mProj = proj();
		// NOTE(jp): Even though GPU will convert it to float, for simplicity we keep this in double for now
		paddedImageData.initZeros(arg.paddedImageSize, arg.paddedImageSize);
		paddedImageData.setXmippOrigin();
		FOR_ALL_ELEMENTS_IN_ARRAY2D(mProj)
			A2D_ELEM(paddedImageData,i,j) = static_cast<float>(A2D_ELEM(mProj, i, j));
		// CenterFFT = center for fft (not fft itself)
		CenterFFT(paddedImageData, true);

		memcpy(outImageData + projectionIndex * outImageDataStride, paddedImageData.data, paddedImageData.getSize() * sizeof(float));

		projectionIndex++;
	}

	amountLoaded.noOfImages = projectionIndex;
}