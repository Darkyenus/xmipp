/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
 * 			   Carlos Oscar S. Sorzano            coss@cnb.csic.es (2016)
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

#ifndef _PROG_DIR_SHARPENING
#define _PROG_DIR_SHARPENING

#include <iostream>
#include <core/xmipp_program.h>
#include <core/xmipp_image.h>
#include <core/metadata.h>
#include <core/xmipp_hdf5.h>
#include <core/xmipp_fft.h>
#include <core/xmipp_fftw.h>
#include <math.h>
#include <limits>
#include <complex>
#include <data/fourier_filter.h>
#include <data/filters.h>
#include <string>
#include "symmetrize.h"
#include "resolution_directional.h"

/**@defgroup Directional Sharpening
   @ingroup ReconsLibrary */
//@{
/** SSNR parameters. */

class ProgDirSharpening : public XmippProgram
{
public:
	 /** Filenames */
	FileName fnOut, fnVol, fnRes, fnMD, fnMask;

	/** sampling rate, minimum resolution, and maximum resolution */
	double sampling, maxRes, minRes, lambda, K, maxFreq, minFreq, desv_Vorig, desvOutside_Vorig, R, significance, res_step;
	int Niter, Nthread;

public:

    void defineParams();
    void readParams();
    void produceSideInfo();
    void icosahedronVertex(Matrix2D<double> &angles);
    void icosahedronFaces(Matrix2D<int> &faces, Matrix2D<double> &vertex);

    void monogenicAmplitude(const MultidimArray< std::complex<double> > &myfftV, MultidimArray<double> &amplitude);

    void defineIcosahedronFaceMask(Matrix2D<int> &faces, Matrix2D<double> &vertex,
    		MultidimArray< std::complex<double> > &myfftV, double &ang_con);

    void localdeblurStep(Matrix2D<int> &faces,
    		MultidimArray< std::complex<double> >  &fftV, MultidimArray<double> &localResolutionMap);

    void faceCenter(int face_number, Matrix2D<int> &faces, Matrix2D<double> &vertex,
    		double &xCenter, double &yCenter, double &zCenter);

    double averageInMultidimArray(MultidimArray<double> &amplitude, MultidimArray<int> &mask);

    void directionalResolutionStep(int face_number, Matrix2D<int> &faces, Matrix2D<double> &vertex,
    		MultidimArray< std::complex<double> > &conefilter, MultidimArray<int> &mask, MultidimArray<double> &localResolutionMap,
    		double &cone_angle);

    void defineIcosahedronCone(int face_number, Matrix2D<int> &faces, Matrix2D<double> &vertex,
    		MultidimArray< std::complex<double> > &myfftV, MultidimArray< std::complex<double> > &conefilter,
			double coneAngle);

    /* Mogonogenid amplitud of a volume, given an input volume,
     * the monogenic amplitud is calculated and low pass filtered at frequency w1*/
    void lowPassFilterFunction(const MultidimArray< std::complex<double> > &myfftV,
    		double w, double wL, MultidimArray<double> &filteredVol, int count);

    void bandPassFilterFunction(const MultidimArray< std::complex<double> > &myfftV,
    		double w, double wL, MultidimArray<double> &filteredVol, int count);

    void wideBandPassFilter(const MultidimArray< std::complex<double> > &myfftV,
                    double wmin, double wmax, double wL, MultidimArray<double> &filteredVol);

      void maxMinResolution(MultidimArray<double> &resVol,
			double &maxRes, double &minRes);

      void computeAvgStdev_within_binary_mask(const MultidimArray< double >&resVol,
      										const MultidimArray< double >&vol, double &stddev, bool outside=false );

    void localDirectionalfiltering(Matrix2D<int> &faces, MultidimArray< std::complex<double> > &myfftV,
            MultidimArray<double> &localfilteredVol,
            double &minRes, double &maxRes, double &step);

    void amplitudeMonogenicSignalBP(MultidimArray< std::complex<double> > &myfftV,
    		double w1, double w1l, MultidimArray<double> &amplitude, int count);

    void sameEnergy(MultidimArray< std::complex<double> > &myfftV,
			MultidimArray<double> &localfilteredVol,
			double &minFreq, double &maxFreq, double &step);

    void run();
public:
    MultidimArray<double> Vorig;//, VsoftMask;
    Image<int> mask;
    MultidimArray<double> resVol;
    MultidimArray<double> iu, VRiesz, sharpenedMap; // Inverse of the frequency
	MultidimArray< std::complex<double> > fftV, fftVfilter, conefilter; // Fourier transform of the input volume
	FourierTransformer transformer, transformer_inv;
	FourierFilter FilterBand;
    MultidimArray< std::complex<double> > fftVRiesz, fftVRiesz_aux;
   	Matrix2D<double> angles, resolutionMatrix, maskMatrix, trigProducts;
	Matrix1D<double> freq_fourier_x, freq_fourier_y, freq_fourier_z;
	int N_smoothing, Rparticle;
	long NVoxelsOriginalMask;
};
//@}
#endif
