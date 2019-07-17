/***************************************************************************
 *
 * Authors:     Jan Polak (456647@mail.muni.cz)
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

Codelets::Codelets()
		: load_projections{0}
		, reconstruct_fft{0}
		, redux_init_volume{0}
		, redux_init_weights{0}
		, redux_sum_volume{0}
		, redux_sum_weights{0} {

	// NOTE(jp): This used to be in designated initializers and this whole cpp wasn't necessary.
	// But C++ does not have them, so it has to be done through this ugly circus.

	static struct starpu_perfmodel load_projections_model;
	load_projections_model.type = STARPU_HISTORY_BASED;
	load_projections_model.symbol = "load_projections_model";


	// Load Projections Codelet
	load_projections.where = STARPU_CPU;
	load_projections.cpu_funcs[0] = func_load_projections;
	load_projections.cpu_funcs_name[0] = "combine_image_func";
	load_projections.nbuffers = 5;
	load_projections.modes[0] = STARPU_W; // LoadProjectionAmountLoaded
	load_projections.modes[1] = STARPU_W; // if hasCTF: CTFs buffer
	load_projections.modes[2] = STARPU_W; // if hasCTF: Modulators Buffer
	load_projections.modes[3] = STARPU_W; // Image Data Buffer
	load_projections.modes[4] = STARPU_W; // Spaces Buffer
	load_projections.name = "codelet_load_projections";
	load_projections.model = &load_projections_model;
	// cl_arg: LoadProjectionArgs - MUST NOT BE COPIED!!!

	/*static struct starpu_perfmodel padded_image_to_fft_model;
	padded_image_to_fft_model.type = STARPU_REGRESSION_BASED;
	padded_image_to_fft_model.symbol = "padded_image_to_fft_model";*/

	// Padded Image to FFT Codelet
	padded_image_to_fft.where = STARPU_CPU | STARPU_CUDA;
	padded_image_to_fft.cpu_funcs[0] = func_padded_image_to_fft_cpu;
	padded_image_to_fft.cpu_funcs_name[0] = "func_padded_image_to_fft_cpu";
	padded_image_to_fft.cuda_funcs[0] = func_padded_image_to_fft_cuda;
	padded_image_to_fft.cuda_flags[0] = STARPU_CUDA_ASYNC;
	padded_image_to_fft.nbuffers = 3;
	padded_image_to_fft.modes[0] = STARPU_R; // Padded Image Data Buffer
	padded_image_to_fft.modes[1] = STARPU_W; // FFT Buffer
	padded_image_to_fft.modes[2] = STARPU_SCRATCH; // Raw FFT Scratch Area
	padded_image_to_fft.name = "codelet_padded_image_to_fft";
	// Not using any model for this, because CUDA codelet has huge initialization time, but then is very fast. This skews all statistics and decisions are not very meaningful.
	//padded_image_to_fft.model = &padded_image_to_fft_model;
	// cl_arg: PaddedImageToFftArgs

	static struct starpu_perfmodel reconstruct_fft_model;
	reconstruct_fft_model.type = STARPU_REGRESSION_BASED;
	reconstruct_fft_model.symbol = "reconstruct_fft_model";

	// Reconstruct FFT Codelet
	reconstruct_fft.where = STARPU_CPU | STARPU_CUDA;
	reconstruct_fft.cpu_funcs[0] = func_reconstruct_cpu_lookup_interpolation;
	reconstruct_fft.cpu_funcs[1] = func_reconstruct_cpu_dynamic_interpolation; // (Typically about 2x slower than lookup interpolation)
	reconstruct_fft.cpu_funcs_name[0] = "func_reconstruct_cpu_lookup_interpolation";
	reconstruct_fft.cpu_funcs_name[1] = "func_reconstruct_cpu_dynamic_interpolation";
	reconstruct_fft.cuda_funcs[0] = func_reconstruct_cuda;
	reconstruct_fft.cuda_flags[0] = STARPU_CUDA_ASYNC;
	reconstruct_fft.nbuffers = 7;
	reconstruct_fft.modes[0] = STARPU_R; // FFT Buffer
	reconstruct_fft.modes[1] = STARPU_R; // CTF Buffer (only if hasCTF)
	reconstruct_fft.modes[2] = STARPU_R; // Modulators Buffer (only if hasCTF)
	reconstruct_fft.modes[3] = STARPU_R; // Traverse Spaces Buffer
	reconstruct_fft.modes[4] = STARPU_R; // Blob Table Squared Buffer (only present if fastLateBlobbing is false)
	reconstruct_fft.modes[5] = STARPU_REDUX; // Result Volume Buffer
	reconstruct_fft.modes[6] = STARPU_REDUX; // Result Weights Buffer
	reconstruct_fft.name = "codelet_reconstruct_fft";
	reconstruct_fft.model = &reconstruct_fft_model;
	// cl_arg: ReconstructFftArgs

	// Redux volume & weights
	// Init volume
	static struct starpu_perfmodel redux_init_volume_model;
	redux_init_volume_model.type = STARPU_REGRESSION_BASED;
	redux_init_volume_model.symbol = "redux_init_volume_model";
	redux_init_volume.where = STARPU_CPU | STARPU_CUDA;
	redux_init_volume.cpu_funcs[0] = func_redux_init_volume_cpu;
	redux_init_volume.cpu_funcs_name[0] = "func_redux_init_volume_cpu";
	redux_init_volume.cuda_funcs[0] = func_redux_init_volume_cuda;
	redux_init_volume.cuda_flags[0] = STARPU_CUDA_ASYNC;
	redux_init_volume.nbuffers = 1;
	redux_init_volume.modes[0] = STARPU_W;
	redux_init_volume.name = "redux_init_volume";
	redux_init_volume.model = &redux_init_volume_model;
	// Init weight
	static struct starpu_perfmodel redux_init_weights_model;
	redux_init_weights_model.type = STARPU_REGRESSION_BASED;
	redux_init_weights_model.symbol = "redux_init_weights_model";
	redux_init_weights.where = STARPU_CPU | STARPU_CUDA;
	redux_init_weights.cpu_funcs[0] = func_redux_init_weights_cpu;
	redux_init_weights.cpu_funcs_name[0] = "func_redux_init_weights_cpu";
	redux_init_weights.cuda_funcs[0] = func_redux_init_weights_cuda;
	redux_init_weights.cuda_flags[0] = STARPU_CUDA_ASYNC;
	redux_init_weights.nbuffers = 1;
	redux_init_weights.modes[0] = STARPU_W;
	redux_init_weights.name = "redux_init_weights";
	redux_init_weights.model = &redux_init_weights_model;
	// Sum volume
	static struct starpu_perfmodel redux_sum_volume_model;
	redux_sum_volume_model.type = STARPU_REGRESSION_BASED;
	redux_sum_volume_model.symbol = "redux_sum_volume_model";
	redux_sum_volume.where = STARPU_CPU | STARPU_CUDA;
	redux_sum_volume.cpu_funcs[0] = func_redux_sum_volume_cpu;
	redux_sum_volume.cpu_funcs_name[0] = "func_redux_sum_volume_cpu";
	redux_sum_volume.cuda_funcs[0] = func_redux_sum_volume_cuda;
	redux_sum_volume.cuda_flags[0] = STARPU_CUDA_ASYNC;
	redux_sum_volume.nbuffers = 2;
	redux_sum_volume.modes[0] = STARPU_RW;
	redux_sum_volume.modes[1] = STARPU_R;
	redux_sum_volume.name = "redux_sum_volume";
	redux_sum_volume.model = &redux_sum_volume_model;
	// Sum weight
	static struct starpu_perfmodel redux_sum_weights_model;
	redux_sum_weights_model.type = STARPU_REGRESSION_BASED;
	redux_sum_weights_model.symbol = "redux_sum_weights_model";
	redux_sum_weights.where = STARPU_CPU | STARPU_CUDA;
	redux_sum_weights.cpu_funcs[0] = func_redux_sum_weights_cpu;
	redux_sum_weights.cpu_funcs_name[0] = "func_redux_sum_weights_cpu";
	redux_sum_weights.cuda_funcs[0] = func_redux_sum_weights_cuda;
	redux_sum_weights.cuda_flags[0] = STARPU_CUDA_ASYNC;
	redux_sum_weights.nbuffers = 2;
	redux_sum_weights.modes[0] = STARPU_RW;
	redux_sum_weights.modes[1] = STARPU_R;
	redux_sum_weights.name = "redux_sum_weights";
	redux_sum_weights.model = &redux_sum_weights_model;
}

Codelets codelet;