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
__kernel void TaskQueue_gpu(__global task_t *ptr_queue, __global int *ptr_data, __global int *consumed, int iterations,
    int offset, int gpuQueueSize, __local task_t *t, __local int *next) {

    const int tid       = get_local_id(0);
    const int tileid    = get_group_id(0);
    int       tile_size = get_local_size(0);

    // Fetch task
    if(tid == 0) {
        *next = atomic_add(consumed, 1);
        t->id = ptr_queue[*next].id;
        t->op = ptr_queue[*next].op;
    }
    barrier(CLK_LOCAL_MEM_FENCE); // It can be removed if work-group = wavefront
    while(*next < gpuQueueSize) {
        // Compute task
        if(t->op == SIGNAL_WORK_KERNEL) {
            for(int i = 0; i < iterations; i++) {
                ptr_data[(t->id - offset) * tile_size + tid] += tile_size;
            }

            ptr_data[(t->id - offset) * tile_size + tid] += t->id;
        }
        if(t->op == SIGNAL_NOTWORK_KERNEL) {
            for(int i = 0; i < 1; i++) {
                ptr_data[(t->id - offset) * tile_size + tid] += tile_size;
            }

            ptr_data[(t->id - offset) * tile_size + tid] += t->id;
        }
        if(tid == 0) {
            *next = atomic_add(consumed, 1);
            // Fetch task
            t->id = ptr_queue[*next].id;
            t->op = ptr_queue[*next].op;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
