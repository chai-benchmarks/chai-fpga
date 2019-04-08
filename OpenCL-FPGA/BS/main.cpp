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

#include "kernel.h"
#include "support/common.h"
#include "support/ocl.h"
#include "support/partitioner.h"
#include "support/timer.h"
#include "support/verify.h"

#include <unistd.h>
#include <thread>

// Params ---------------------------------------------------------------------
struct Params {

    int         platform;
    int         device;
    int         n_work_items;
    int         n_work_groups;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    float       alpha;
    const char *file_name;
    int         in_size_i;
    int         in_size_j;
    int         out_size_i;
    int         out_size_j;

    Params(int argc, char **argv) {
        platform      = 0;
        device        = 0;
        n_work_items  = 16;
        n_work_groups = 32;
        n_threads     = 4;
        n_warmup      = 5;
        n_reps        = 50;
        alpha         = 0.1;
        file_name     = "input/control.txt";
        in_size_i = in_size_j = 3;
        out_size_i = out_size_j = 300;
        char opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:a:f:m:n:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'p': platform      = atoi(optarg); break;
            case 'd': device        = atoi(optarg); break;
            case 'i': n_work_items  = atoi(optarg); break;
            case 'g': n_work_groups = atoi(optarg); break;
            case 't': n_threads     = atoi(optarg); break;
            case 'w': n_warmup      = atoi(optarg); break;
            case 'r': n_reps        = atoi(optarg); break;
            case 'a': alpha         = atof(optarg); break;
            case 'f': file_name     = optarg; break;
            case 'm': in_size_i = in_size_j = atoi(optarg); break;
            case 'n': out_size_i = out_size_j = atoi(optarg); break;
            default:
                cerr << "\nUnrecognized option!" << endl;
                usage();
                exit(0);
            }
        }
    }

    void usage() {
        cerr << "\nUsage:  ./bs [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -p <P>    OpenCL platform ID (default=0)"
                "\n    -d <D>    OpenCL device ID (default=0)"
                "\n    -i <I>    # of device work-items"
                "\n    -g <G>    # of device work-groups"
                "\n    -t <T>    # of host threads"
                "\n    -w <W>    # of untimed warmup iterations"
                "\n    -r <R>    # of timed repition iterations"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of output elements to process on host"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    name of input file with control points"
                "\n    -m <N>    input size in both dimensions"
                "\n    -n <R>    output resolution in both dimensions"
                "\n";
    }
};

// Input Data -----------------------------------------------------------------
void read_input(XYZ *in, const Params &p) {

    // Open input file
    FILE *f = NULL;
    f       = fopen(p.file_name, "r");
    if(f == NULL) {
        puts("Error opening file");
        exit(-1);
    }

    // Store points from input file to array
    int k = 0, ic = 0;
    XYZ v[10000];
#if DOUBLE_PRECISION
    while(fscanf(f, "%lf,%lf,%lf", &v[ic].x, &v[ic].y, &v[ic].z) == 3)
#else
    while(fscanf(f, "%f,%f,%f", &v[ic].x, &v[ic].y, &v[ic].z) == 3)
#endif
    {
        ic++;
    }
    for(int i = 0; i <= p.in_size_i; i++) {
        for(int j = 0; j <= p.in_size_j; j++) {
            in[i * (p.in_size_j + 1) + j].x = v[k].x;
            in[i * (p.in_size_j + 1) + j].y = v[k].y;
            in[i * (p.in_size_j + 1) + j].z = v[k].z;
            //k++;
            k = (k + 1) % 16;
        }
    }
}

// Main -----------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    OpenCLSetup  ocl(p.platform, p.device);
    Timer        timer;
    cl_int       clStatus;

    // Allocate
    timer.start("Allocation");
    unsigned int in_size  = (p.in_size_i + 1) * (p.in_size_j + 1) * sizeof(XYZ);
    unsigned int out_size = p.out_size_i * p.out_size_j * sizeof(XYZ);
#ifdef OCL_2_0
    XYZ *            h_in     = (XYZ *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, in_size, 0);
    XYZ *            h_out    = (XYZ *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, out_size, 0);
    XYZ *            d_in     = h_in;
    XYZ *            d_out    = h_out;
    std::atomic_int *worklist = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int), 0);
#else
    XYZ *  h_in        = (XYZ *)malloc(in_size);
    XYZ *  h_out       = (XYZ *)malloc(out_size);
    XYZ *  h_out_merge = (XYZ *)malloc(out_size);
    cl_mem d_in  = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, in_size, NULL, &clStatus);
    cl_mem d_out = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, out_size, NULL, &clStatus);
#endif
    clFinish(ocl.clCommandQueue);
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    read_input(h_in, p);
    unsigned int n_tasks_i   = divceil(p.out_size_i, p.n_work_items);
    unsigned int n_tasks_j   = divceil(p.out_size_j, p.n_work_items);
    Partitioner  partitioner = partitioner_create(n_tasks_i * n_tasks_j, p.alpha);
    clFinish(ocl.clCommandQueue);
    timer.stop("Initialization");
    timer.print("Initialization", 1);

#ifndef OCL_2_0
    // Copy to device
    timer.start("Copy To Device");
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_in, CL_TRUE, 0, in_size, h_in, 0, NULL, NULL);
    clFinish(ocl.clCommandQueue);
    CL_ERR();
    timer.stop("Copy To Device");
    timer.print("Copy To Device", 1);
#endif

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; ++rep) {

// Reset
#ifdef OCL_2_0
        if(partitioner.strategy == DYNAMIC_PARTITIONING) {
            worklist[0].store(0);
        }
#endif

        if(rep >= p.n_warmup)
            timer.start("Kernel");

// Launch GPU threads
#ifdef OCL_2_0
        clSetKernelArgSVMPointer(ocl.clKernel, 0, d_in);
        clSetKernelArgSVMPointer(ocl.clKernel, 1, d_out);
#else
        clSetKernelArg(ocl.clKernel, 0, sizeof(cl_mem), &d_in);
        clSetKernelArg(ocl.clKernel, 1, sizeof(cl_mem), &d_out);
#endif
   
	#ifdef C_PLUS
        clSetKernelArg(ocl.clKernel, 2, sizeof(Partitioner), &partitioner);
#else
        // Pointer to partitioner
        cl_mem d_partitioner = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(Partitioner), NULL, &clStatus);
        clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_partitioner, CL_TRUE, 0, sizeof(Partitioner), &partitioner, 0, NULL, NULL);
        clSetKernelArg(ocl.clKernel, 2, sizeof(cl_mem), &d_partitioner);
#endif


        clSetKernelArg(ocl.clKernel, 3, sizeof(int), &p.in_size_i);
        clSetKernelArg(ocl.clKernel, 4, sizeof(int), &p.in_size_j);
        clSetKernelArg(ocl.clKernel, 5, sizeof(int), &p.out_size_i);
        clSetKernelArg(ocl.clKernel, 6, sizeof(int), &p.out_size_j);
#ifdef OCL_2_0
        clSetKernelArgSVMPointer(ocl.clKernel, 9, worklist);
#endif
        // Kernel launch
        size_t ls[2] = {(size_t)p.n_work_items, (size_t)p.n_work_items};
        size_t gs[2] = {(size_t)p.n_work_items * p.n_work_groups, (size_t)p.n_work_items};
        if(p.n_work_groups > 0) {
            clStatus = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 2, NULL, gs, ls, 0, NULL, NULL);
            CL_ERR();
        }

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_in, h_out, partitioner, p.n_threads, p.n_work_items, p.in_size_i,
            p.in_size_j, p.out_size_i, p.out_size_j
#ifdef OCL_2_0
            ,
            worklist
#endif
            );

        clFinish(ocl.clCommandQueue);
        main_thread.join();

        if(rep >= p.n_warmup)
            timer.stop("Kernel");
	
	#ifndef C_PLUS
        // Free d_partitioner
           clReleaseMemObject(d_partitioner);
        #endif
    }
    timer.print("Kernel", p.n_reps);

#ifndef OCL_2_0
    // Copy back
    timer.start("Copy Back and Merge");
    clStatus = clEnqueueReadBuffer(ocl.clCommandQueue, d_out, CL_TRUE, 0, out_size, h_out_merge, 0, NULL, NULL);
    CL_ERR();
    clFinish(ocl.clCommandQueue);
    // Merge
    for(unsigned int t = 0; t < partitioner.cut; ++t) {
        const int ty  = t / n_tasks_j;
        const int tx  = t % n_tasks_j;
        int       row = ty * p.n_work_items;
        int       col = tx * p.n_work_items;
        for(int i = row; i < row + p.n_work_items; ++i) {
            for(int j = col; j < col + p.n_work_items; ++j) {
                if(i < p.out_size_i && j < p.out_size_j) {
                    h_out_merge[i * p.out_size_j + j] = h_out[i * p.out_size_j + j];
                }
            }
        }
    }
    timer.stop("Copy Back and Merge");
    timer.print("Copy Back and Merge", 1);
#endif

// Verify answer
#ifdef OCL_2_0
    verify(h_in, h_out, p.in_size_i, p.in_size_j, p.out_size_i, p.out_size_j);
#else
    verify(h_in, h_out_merge, p.in_size_i, p.in_size_j, p.out_size_i, p.out_size_j);
#endif

    // Free memory
    timer.start("Deallocation");
#ifdef OCL_2_0
    clSVMFree(ocl.clContext, h_in);
    clSVMFree(ocl.clContext, h_out);
    clSVMFree(ocl.clContext, worklist);
#else
    free(h_in);
    free(h_out);
    free(h_out_merge);
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
#endif
    ocl.release();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    return 1;
}
