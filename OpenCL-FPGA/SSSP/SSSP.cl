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
__kernel void SSSP_gpu(__global Node *graph_nodes_av, __global Edge *graph_edges_av, __global int *ptr_cost,
    __global int *ptr_color, __global int *ptr_q1, __global int *ptr_q2, __global int *ptr_num_t,
    __global int *ptr_head, __global int *ptr_tail, __global int *ptr_threads_end, __global int *ptr_threads_run,
    __global int *ptr_overflow, __global int *ptr_gray_shade, __global int *ptr_iter, __local int *tail_bin,
    __local int *elems, __local int *shift, __local int *base, int LIMIT, const int CPU) {

    const int tid     = get_local_id(0);
    const int gtid    = get_global_id(0);
    const int MAXWG   = get_num_groups(0);
    const int WG_SIZE = get_local_size(0);

    int iter = atomic_add(&ptr_iter[0], 0);

    int num_t_local = atomic_add(ptr_num_t, 0);

    int gray_shade = atomic_add(&ptr_gray_shade[0], 0);

    if(tid == 0) {
        // Reset queue
        *tail_bin = 0;
    }

    // Fetch frontier elements from the queue
    if(tid == 0)
        *base = atomic_add(&ptr_head[0], WG_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);

    int my_base = *base;
    while(my_base < num_t_local) {

        // If local queue might overflow
        if(*tail_bin >= W_QUEUE_SIZE / 2) {
            if(tid == 0) {
                // Add local tail_bin to ptr_tail
                *shift = atomic_add(&ptr_tail[0], *tail_bin);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            int local_shift = tid;
            while(local_shift < *tail_bin) {
                ptr_q2[*shift + local_shift] = elems[local_shift];
                // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
                local_shift += WG_SIZE;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if(tid == 0) {
                // Reset local queue
                *tail_bin = 0;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(my_base + tid < num_t_local && *ptr_overflow == 0) {
            // Visit a node from the current frontier
            int pid = ptr_q1[my_base + tid];
            //////////////// Visit node ///////////////////////////
            atomic_xchg(&ptr_color[pid], BLACK); // Node visited
            int  cur_cost = atomic_add(&ptr_cost[pid], 0); // Look up shortest-path distance to this node
            Node cur_node;
            cur_node.x = graph_nodes_av[pid].x;
            cur_node.y = graph_nodes_av[pid].y;
            Edge cur_edge;
            // For each outgoing edge
            for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
                cur_edge.x = graph_edges_av[i].x;
                cur_edge.y = graph_edges_av[i].y;
                int id     = cur_edge.x;
                int cost   = cur_edge.y;
                cost += cur_cost;
                int orig_cost = atomic_max(&ptr_cost[id], cost);
                if(orig_cost < cost) {
                    int old_color = atomic_max(&ptr_color[id], gray_shade);
                    if(old_color != gray_shade) {
                        // Push to the queue
                        int tail_index = atomic_add(tail_bin, 1);
                        if(tail_index >= W_QUEUE_SIZE) {
                            *ptr_overflow = 1;
                        } else
                            elems[tail_index] = id;
                    }
                }
            }
        }

        if(tid == 0)
            *base = atomic_add(&ptr_head[0], WG_SIZE); // Fetch more frontier elements from the queue
        barrier(CLK_LOCAL_MEM_FENCE);
        my_base = *base;
    }

    printf("%d done checkpoint 1\n", tid);
    /////////////////////////////////////////////////////////
    // Compute size of the output and allocate space in the global queue
    if(tid == 0) {
        *shift = atomic_add(&ptr_tail[0], *tail_bin);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ///////////////////// CONCATENATE INTO HOST COHERENT MEMORY /////////////////////
    int local_shift = tid;
    while(local_shift < *tail_bin) {
        ptr_q2[*shift + local_shift] = elems[local_shift];
        // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
        local_shift += WG_SIZE;
    }
    //////////////////////////////////////////////////////////////////////////
    printf("%d done checkpoint 2\n", tid);
    if(gtid == 0) {
    printf("%d atomic_add\n", tid);
        atomic_add(&ptr_iter[0], 1);
    }
}
