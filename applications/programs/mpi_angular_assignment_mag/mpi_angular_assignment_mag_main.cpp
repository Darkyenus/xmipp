/***************************************************************************
 *
 * Authors:  Jeison Méndez García jmendez@utp.edu.co
 *
 * Universidad Nacional Autónoma de México
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
 *  e-mail address 'xmipp@cnb.uam.es'
 ***************************************************************************/

#include <mpi.h>
#include <parallel/xmipp_mpi.h>
#include <reconstruction/angular_assignment_mag.h>

 /*
#include <../parallel/xmipp_mpi.h>
#include <../reconstruction/angular_assignment_mag.h> // */

//CREATE_MPI_METADATA_PROGRAM(ProgAngularAssignmentMag, MpiProgAngularAssignmentMag)
// /*
class MpiProgAngularAssignmentMag: public ProgAngularAssignmentMag, public MpiMetadataProgram
{
public:
    int Nsimul;

    void defineParams()
    {
        ProgAngularAssignmentMag::defineParams();
        MpiMetadataProgram::defineParams();
        addParamsLine("  [--Nsimultaneous <N=1>]     : Number of simultaneous processes that can enter in preprocessing");
    }
    void readParams()
    {
        MpiMetadataProgram::readParams();
        ProgAngularAssignmentMag::readParams();
        Nsimul = getIntParam("--Nsimultaneous");
    }
    void read(int argc, char **argv, bool reportErrors = true)
    {
        MpiMetadataProgram::read(argc,argv);
    }
    void preProcess()
    {
        int Nturns = (int)ceil(node->size/Nsimul);
        int myTurn = (int)floor(node->rank/Nsimul);
        for (int turn=0; turn<=Nturns; turn++)
        {
            if (turn==myTurn)
                ProgAngularAssignmentMag::preProcess();
            node->barrierWait();
        }
        MetaData &mdIn = *getInputMd();
        mdIn.addLabel(MDL_GATHER_ID);
        mdIn.fillLinear(MDL_GATHER_ID,1,1);
        createTaskDistributor(mdIn, blockSize);
    }
    void startProcessing()
    {
        if (node->rank==1)
        {
        	verbose=1;
            ProgAngularAssignmentMag::startProcessing();
        }
        node->barrierWait();
    }
    void showProgress()
    {
        if (node->rank==1)
        {
            time_bar_done=first+1;
            ProgAngularAssignmentMag::showProgress();
        }
    }
    bool getImageToProcess(size_t &objId, size_t &objIndex)
    {
        return getTaskToProcess(objId, objIndex);
    }
    void finishProcessing()
    {
        node->gatherMetadatas(*getOutputMd(), fn_out);
    	MetaData MDaux;
    	MDaux.sort(*getOutputMd(), MDL_GATHER_ID);
        MDaux.removeLabel(MDL_GATHER_ID);
        *getOutputMd()=MDaux;
        if (node->isMaster())
            ProgAngularAssignmentMag::finishProcessing();
    }
    void wait()
    {
		distributor->wait();
    }
}; // */

RUN_XMIPP_PROGRAM(MpiProgAngularAssignmentMag)
