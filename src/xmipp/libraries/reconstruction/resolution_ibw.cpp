/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *              Alvaro Capell
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
#include "resolution_ibw.h"
#include <data/filters.h>
#include <data/morphology.h>

/* Read parameters --------------------------------------------------------- */
void ProgResolutionIBW::readParams()
{
    fnVol = getParam("-i");
    fnOut = getParam("-o");
}

/* Usage ------------------------------------------------------------------- */
void ProgResolutionIBW::defineParams()
{
    addUsageLine("Evaluate the resolution of a volume through the inverse border widths");
    addParamsLine("   -i <file>              : Volume to evaluate");
    addParamsLine("  [-o <file=\"\">]        : Volume with the border widths of the edge voxels");
}

/* Show -------------------------------------------------------------------- */
void ProgResolutionIBW::show() const
{
	if (verbose==0)
		return;
    std::cout << "Input volume:      " << fnVol << std::endl
    		  << "Output widths:     " << fnOut << std::endl
    ;
}

/* Run --------------------------------------------------------------------- */
//#define DEBUG
void ProgResolutionIBW::run()
{
    V.read(fnVol);

    //Mask generation
    Image<double> aux;
    double bg_mean;
    MultidimArray<double> Vmask;
    detectBackground(V(),aux(),0.1,bg_mean);
#ifdef DEBUG

    aux.write("PPPmask_no_ero_03.vol");
#endif

    //Mask volume erosion to expand the mask boundaries
    Vmask.initZeros(V());
    erode3D(aux(),Vmask, 18,0,2);

    //Correction of some flaws produced in the edges of the mask volume
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Vmask)
    if (k<=4 || i<=4 || j<=4 ||
        k>=ZSIZE(Vmask)-4 || i>=YSIZE(Vmask)-4 || j>=XSIZE(Vmask)-4)
        DIRECT_A3D_ELEM(Vmask,k,i,j)=1;

    aux()=Vmask;
#ifdef DEBUG

    aux.write("PPPmask_ero_03.vol");
#endif

    //Sobel edge detection applied to original volume
    Image<double> Vedge;
    computeEdges(V(),Vedge());
#ifdef DEBUG

    Vedge.write("PPPvolume_sobel_unmask_03.vol");
#endif

    //Masked volume generation
    const MultidimArray<double> &mVedge=Vedge();
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mVedge)
    if (DIRECT_MULTIDIM_ELEM(Vmask,n)==1)
        DIRECT_MULTIDIM_ELEM(mVedge,n)=0;
#ifdef DEBUG

    Vedge.write("volume_sobel_mask_03.vol");
#endif

    double minval, maxval, avg, stddev;

    //Invert the mask to meet computeStats_within_binary_mask requirements
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Vmask)
    if (DIRECT_MULTIDIM_ELEM(Vmask,n)==1)
        DIRECT_MULTIDIM_ELEM(Vmask,n)=0;
    else
        DIRECT_MULTIDIM_ELEM(Vmask,n)=1;

    //Threshold is 3 times the standard deviation of unmasked pixel values
    double thresh;
    computeStats_within_binary_mask(Vmask,mVedge,minval, maxval, avg, stddev);
    thresh=3*stddev;

    //Final edge volume generated by setting to 1 positions with values > threshold
    Image<double> Vaux;
    Vaux().initZeros(mVedge);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mVedge)
    if (DIRECT_MULTIDIM_ELEM(mVedge,n)>=thresh)
        DIRECT_MULTIDIM_ELEM(Vaux(),n)=1;

#ifdef DEBUG

    Vaux.write("volumen_bordes_definitivo_03.vol");
#endif

    const MultidimArray<double> &mVaux=Vaux();

    //Spline coefficient volume from original volume, to allow <1 step sizes
    MultidimArray<double> Volcoeffs;
    Volcoeffs.initZeros(V());

    produceSplineCoefficients(3,Volcoeffs,V());

    //Width parameter volume initialization
    Image<double> widths;
    widths().resizeNoCopy(V());
    widths().initConstant(1e5);
    double step=0.25;

    Matrix1D<double> direction(3);

    //Calculation of edge width for 10 different directions, if a smaller value is found for a different
    //direction on a given position the former value is overwritten

    //Direction (1,0,0)
    VECTOR_R3(direction,1,0,0);
    edgeWidth(Volcoeffs, mVaux, widths(), direction, step);

    //Direction (0,1,0)
    VECTOR_R3(direction,0,1,0);
    edgeWidth(Volcoeffs, mVaux, widths(), direction, step);

    //Direction (0,0,1)
    VECTOR_R3(direction,0,0,1);
    edgeWidth(Volcoeffs, mVaux, widths(), direction, step);

    //Direction (1,1,0)
    VECTOR_R3(direction,(1/sqrt(2)),(1/sqrt(2)),0);
    edgeWidth(Volcoeffs, mVaux, widths(), direction, step);

    //Direction (1,0,1)
    VECTOR_R3(direction,(1/sqrt(2)),0,(1/sqrt(2)));
    edgeWidth(Volcoeffs, mVaux, widths(), direction, step);

    //Direction (0,1,1)
    VECTOR_R3(direction,0,(1/sqrt(2)),(1/sqrt(2)));
    edgeWidth(Volcoeffs, mVaux, widths(), direction, step);

    //Direction (1,1,1)
    VECTOR_R3(direction,(1/sqrt(3)),(1/sqrt(3)),(1/sqrt(3)));
    edgeWidth(Volcoeffs, mVaux, widths(), direction, step);

    //Direction (-1,1,1)
    VECTOR_R3(direction,-(1/sqrt(3)),(1/sqrt(3)),(1/sqrt(3)));
    edgeWidth(Volcoeffs, mVaux, widths(), direction, step);

    //Direction (1,1,-1)
    VECTOR_R3(direction,(1/sqrt(3)),(1/sqrt(3)),-(1/sqrt(3)));
    edgeWidth(Volcoeffs, mVaux, widths(), direction, step);

    //Direction (1,-1,1)
    VECTOR_R3(direction,(1/sqrt(3)),-(1/sqrt(3)),(1/sqrt(3)));
    edgeWidth(Volcoeffs, mVaux, widths(), direction, step);

#ifdef DEBUG

    std::cout << "width stats: ";
    widths().printStats();
    std::cout << std::endl;
    widths.write("PPPwidths.vol");
#endif

    double ibw=calculateIBW(widths());
    std::cout << "Resolution ibw= " << ibw << std::endl;
    if (fnOut!="")
    	widths.write(fnOut);
}

void ProgResolutionIBW::edgeWidth(const MultidimArray<double> &volCoeffs,
                                  const MultidimArray<double> &edges,
                                  MultidimArray <double>& widths, const Matrix1D<double> &dir,
                                  double step) const
{
    double forward_count, backward_count, slope;
    Matrix1D<double> pos_aux_fw(3), pos_aux_bw(3), pos(3), pos_aux(3), next_pos(3), Kdir;

    Kdir=step*dir;

    //Visit all elements in volume
    FOR_ALL_ELEMENTS_IN_ARRAY3D(edges)
    {
        //Check for border pixels
        if (A3D_ELEM(edges,k,i,j)!=0)
        {
            //reset all counters
            forward_count=0;
            backward_count=0;
            VECTOR_R3(pos_aux_fw,j,i,k);
            pos_aux_bw=pos=pos_aux_fw;

            //find out if pixel magnitude grows or decreases
            pos_aux=pos;
            pos_aux+=dir;
            double value_plus_dir=volCoeffs.interpolatedElementBSpline3D(XX(pos_aux),YY(pos_aux),ZZ(pos_aux));

            pos_aux=pos;
            pos_aux-=dir;
            double value_minus_dir=volCoeffs.interpolatedElementBSpline3D(XX(pos_aux),YY(pos_aux),ZZ(pos_aux));

            slope=value_plus_dir-value_minus_dir;

            double sign;
            if (slope>0)
                sign=1;
            else
                sign=-1;

            //current_pixel is multiplied by the sign, so only one condition is enough to detect an
            //extremum no matter if the pixel values increase or decrease
            double current_pixel=sign*volCoeffs.interpolatedElementBSpline3D
                                 (XX(pos_aux_fw),YY(pos_aux_fw),ZZ(pos_aux_fw));

            double next_pixel;
            bool not_found;

            //Search for local extremum ahead of the edge in the given direction
            do
            {
                not_found=true;
                next_pos=pos_aux_fw+Kdir;
                next_pixel=sign*volCoeffs.interpolatedElementBSpline3D
                           (XX(next_pos),YY(next_pos),ZZ(next_pos));

                if(next_pixel>current_pixel)
                {
                    current_pixel=next_pixel;
                    pos_aux_fw=next_pos;
                    forward_count++;
                }
                else
                {
                    not_found=false;
                }
            }
            while(not_found);

            current_pixel=sign*volCoeffs.interpolatedElementBSpline3D
                          (XX(pos_aux_bw),YY(pos_aux_bw),ZZ(pos_aux_bw));

            //Search for local extremum behind of the edge in the given direction
            do
            {
                not_found=true;
                next_pos=pos_aux_bw-Kdir;
                next_pixel=sign*volCoeffs.interpolatedElementBSpline3D
                           (XX(next_pos),YY(next_pos),ZZ(next_pos));

                if(next_pixel<current_pixel)
                {
                    current_pixel=next_pixel;
                    pos_aux_bw=next_pos;
                    backward_count++;
                }
                else
                {
                    not_found=false;
                }
            }
            while(not_found);

            //If the width found for this position is smaller than the one stores in edges volume
            //before it is overwritten
            if ((forward_count+backward_count)<A3D_ELEM(widths,k,i,j))
            {
                A3D_ELEM(widths,k,i,j)=forward_count+backward_count;
            }
        }
    }
}

double ProgResolutionIBW::calculateIBW(MultidimArray <double>& widths) const
{
    double total, count;
    total=count=0;

    FOR_ALL_ELEMENTS_IN_ARRAY3D(widths)
    {
    	double width=A3D_ELEM(widths,k,i,j);
        if (width<1e4)
        {
            total+=width;
            count++;
        }
    }
    double avg = total/count;
    return 1.0/avg;
}
