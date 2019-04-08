/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#define _OPENCL_COMPILER_

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#include "support/common.h"

// OpenCL kernel ------------------------------------------------------------------------------------------
__kernel void Histogram_kernel(
#ifdef FPGA
   __global unsigned int* restrict histo, __global unsigned int* restrict data,
#else
    __global unsigned int *histo, __global unsigned int *data, 
#endif
    int size, int bins, int cpu_bins
#ifdef OCL_2_0
    ,
    __local atomic_uint *l_histo
#else
    ,
    __local unsigned int *l_histo
#endif
    ) {
    // Block and thread index
    const int bx = get_group_id(0);
    const int tx = get_local_id(0);
    const int bD = get_local_size(0);
    const int gD = get_num_groups(0);

    // Output partition
    int bins_per_wg   = (bins - cpu_bins) / gD;
    int my_bins_start = bx * bins_per_wg + cpu_bins;
    int my_bins_end   = my_bins_start + bins_per_wg;

    // Constants for read access
    const int begin = tx;
    const int end   = size;
    const int step  = bD;

    // Sub-histograms initialization
    for(int pos = tx; pos < bins_per_wg; pos += bD) {
#ifdef OCL_2_0
        atomic_store(&l_histo[pos], 0);
#else
        l_histo[pos] = 0;
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE); // Intra-block synchronization

    // Main loop
    for(int i = begin; i < end; i += step) {
        // Global memory read
        unsigned int d = ((data[i] * bins) >> 12);

        if(d >= my_bins_start && d < my_bins_end) {
// Atomic vote in shared memory
#ifdef OCL_2_0
            atomic_fetch_add(&l_histo[d - my_bins_start], 1);
#else
            atomic_add(&l_histo[d - my_bins_start], 1);
#endif
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE); // Intra-block synchronization

    // Merge per-block histograms and write to global memory
    for(int pos = tx; pos < bins_per_wg; pos += bD) {
        unsigned int sum = 0;
        for(int base = 0; base < (bins_per_wg); base += (bins_per_wg))
#ifdef OCL_2_0
            sum += (unsigned int)atomic_load(&l_histo[base + pos]);
#else
            sum += l_histo[base + pos];
#endif
        // Atomic addition in global memory
        histo[pos + my_bins_start] += sum;
    }
}
