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

#include <string.h>
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
    int         max_iter;
    int         error_threshold;
    float       convergence_threshold;

    Params(int argc, char **argv) {
        platform              = 0;
        device                = 0;
        n_work_items          = 256;
        n_work_groups         = 8;
        n_threads             = 4;
        n_warmup              = 5;
        n_reps                = 50;
        alpha                 = 0.2;
        file_name             = "input/vectors.csv";
        max_iter              = 2000;
        error_threshold       = 3;
        convergence_threshold = 0.75;
        char opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:a:f:m:e:c:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'p': platform              = atoi(optarg); break;
            case 'd': device                = atoi(optarg); break;
            case 'i': n_work_items          = atoi(optarg); break;
            case 'g': n_work_groups         = atoi(optarg); break;
            case 't': n_threads             = atoi(optarg); break;
            case 'w': n_warmup              = atoi(optarg); break;
            case 'r': n_reps                = atoi(optarg); break;
            case 'a': alpha                 = atof(optarg); break;
            case 'f': file_name             = optarg; break;
            case 'm': max_iter              = atoi(optarg); break;
            case 'e': error_threshold       = atoi(optarg); break;
            case 'c': convergence_threshold = atof(optarg); break;
            default:
                cerr << "\nUnrecognized option!" << endl;
                usage();
                exit(0);
            }
        }
    }

    void usage() {
        cerr << "\nUsage:  ./rscd [options]"
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
                "\n    -a <A>    fraction of input elements to process on host"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    input file name"
                "\n    -m <M>    maximum # of iterations"
                "\n    -e <E>    error threshold"
                "\n    -c <C>    convergence threshold"
                "\n";
    }
};

// Input ----------------------------------------------------------------------
int read_input_size(const Params &p) {
    FILE *File = NULL;
    File       = fopen(p.file_name, "r");
    if(File == NULL) {
        puts("Error al abrir el fichero");
        exit(-1);
    }

    int n;
    fscanf(File, "%d", &n);

    fclose(File);

    return n;
}

void read_input(flowvector *v, int *r, const Params &p) {

    int ic = 0;

    // Open input file
    FILE *File = NULL;
    File       = fopen(p.file_name, "r");
    if(File == NULL) {
        puts("Error opening file!");
        exit(-1);
    }

    int n;
    fscanf(File, "%d", &n);

    while(fscanf(File, "%d,%d,%d,%d", &v[ic].x, &v[ic].y, &v[ic].vx, &v[ic].vy) == 4) {
        ic++;
        if(ic > n) {
            puts("Error: inconsistent file data!");
            exit(-1);
        }
    }
    if(ic < n) {
        puts("Error: inconsistent file data!");
        exit(-1);
    }

    srand(time(NULL));
    for(int i = 0; i < 2 * p.max_iter; i++) {
        r[i] = ((int)rand()) % n;
    }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    OpenCLSetup  ocl(p.platform, p.device);
    Timer        timer;
    cl_int       clStatus;

    // Allocate
    timer.start("Allocation");
    int n_flow_vectors = read_input_size(p);
    int best_model     = -1;
    int best_outliers  = n_flow_vectors;
#ifdef OCL_2_0
    flowvector *h_flow_vector_array =
        (flowvector *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, n_flow_vectors * sizeof(flowvector), 0);
    int *h_random_numbers =
        (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, 2 * p.max_iter * sizeof(int), 0);
    int *h_model_candidate =
        (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, p.max_iter * sizeof(int), 0);
    int *h_outliers_candidate =
        (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, p.max_iter * sizeof(int), 0);
    float *h_model_param_local =
        (float *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, 4 * p.max_iter * sizeof(float), 0);
    std::atomic_int *h_g_out_id = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int), 0);
    flowvector *     d_flow_vector_array  = h_flow_vector_array;
    int *            d_random_numbers     = h_random_numbers;
    int *            d_model_candidate    = h_model_candidate;
    int *            d_outliers_candidate = h_outliers_candidate;
    float *          d_model_param_local  = h_model_param_local;
    std::atomic_int *d_g_out_id           = h_g_out_id;
    std::atomic_int *worklist             = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int), 0);
#else
    flowvector *     h_flow_vector_array  = (flowvector *)malloc(n_flow_vectors * sizeof(flowvector));
    int *            h_random_numbers     = (int *)malloc(2 * p.max_iter * sizeof(int));
    int *            h_model_candidate    = (int *)malloc(p.max_iter * sizeof(int));
    int *            h_outliers_candidate = (int *)malloc(p.max_iter * sizeof(int));
    float *          h_model_param_local  = (float *)malloc(4 * p.max_iter * sizeof(float));
    std::atomic_int *h_g_out_id           = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    cl_mem           d_flow_vector_array  = clCreateBuffer(
        ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, n_flow_vectors * sizeof(flowvector), NULL, &clStatus);
    cl_mem d_random_numbers = clCreateBuffer(
        ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 2 * p.max_iter * sizeof(int), NULL, &clStatus);
    cl_mem d_model_candidate = clCreateBuffer(
        ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, p.max_iter * sizeof(int), NULL, &clStatus);
    cl_mem d_outliers_candidate = clCreateBuffer(
        ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, p.max_iter * sizeof(int), NULL, &clStatus);
    cl_mem d_model_param_local = clCreateBuffer(
        ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 4 * p.max_iter * sizeof(float), NULL, &clStatus);
    cl_mem d_g_out_id =
      clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int), NULL, &clStatus);
#ifdef FPGA
    cl_mem d_partitioner = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(Partitioner), NULL, &clStatus);
#endif
    CL_ERR();
#endif
    clFinish(ocl.clCommandQueue);
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    read_input(h_flow_vector_array, h_random_numbers, p);
    Partitioner partitioner = partitioner_create(p.max_iter, p.alpha);
    clFinish(ocl.clCommandQueue);
    timer.stop("Initialization");
    timer.print("Initialization", 1);

#ifndef OCL_2_0
    // Copy to device
    timer.start("Copy To Device");
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_flow_vector_array, CL_TRUE, 0,
        n_flow_vectors * sizeof(flowvector), h_flow_vector_array, 0, NULL, NULL);
//    std::cout << "size: " << n_flow_vectors * sizeof(flowvector) << std::endl;
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_random_numbers, CL_TRUE, 0, 2 * p.max_iter * sizeof(int),
        h_random_numbers, 0, NULL, NULL);
//    std::cout << "size: " << p.max_iter * sizeof(int) << std::endl; 
    clStatus = clEnqueueWriteBuffer(
        ocl.clCommandQueue, d_model_candidate, CL_TRUE, 0, p.max_iter * sizeof(int), h_model_candidate, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_outliers_candidate, CL_TRUE, 0, p.max_iter * sizeof(int),
        h_outliers_candidate, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_model_param_local, CL_TRUE, 0, 4 * p.max_iter * sizeof(float),
        h_model_param_local, 0, NULL, NULL);
//    std::cout << "size: " << p.max_iter * sizeof(float) << std::endl;
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_g_out_id, CL_TRUE, 0, sizeof(int), h_g_out_id, 0, NULL, NULL);
#ifdef FPGA
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_partitioner, CL_TRUE, 0, sizeof(Partitioner), &partitioner, 0, NULL, NULL);
//    std::cout << "size: " << sizeof(Partitioner) << std::endl;
#endif
    clFinish(ocl.clCommandQueue);
    CL_ERR();
    timer.stop("Copy To Device");
    timer.print("Copy To Device", 1);
#endif

    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Reset
        memset((void *)h_model_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)h_outliers_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)h_model_param_local, 0, 4 * p.max_iter * sizeof(float));
#ifdef OCL_2_0
        h_g_out_id[0].store(0);
        if(partitioner.strategy == DYNAMIC_PARTITIONING) {
            worklist[0].store(0);
        }
#else
        h_g_out_id[0] = 0;
        clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_model_candidate, CL_TRUE, 0, p.max_iter * sizeof(int),
            h_model_candidate, 0, NULL, NULL);
        clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_outliers_candidate, CL_TRUE, 0, p.max_iter * sizeof(int),
            h_outliers_candidate, 0, NULL, NULL);
        clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_model_param_local, CL_TRUE, 0,
            4 * p.max_iter * sizeof(float), h_model_param_local, 0, NULL, NULL);
        clStatus =
            clEnqueueWriteBuffer(ocl.clCommandQueue, d_g_out_id, CL_TRUE, 0, sizeof(int), h_g_out_id, 0, NULL, NULL);
        CL_ERR();
#endif
        clFinish(ocl.clCommandQueue);

        if(rep >= p.n_warmup)
            timer.start("Kernel");

// Launch GPU threads
#ifdef OCL_2_0
        clSetKernelArgSVMPointer(ocl.clKernel, 0, d_model_param_local);
        clSetKernelArgSVMPointer(ocl.clKernel, 1, d_flow_vector_array);
#else
        clSetKernelArg(ocl.clKernel, 0, sizeof(cl_mem), &d_model_param_local);
        clSetKernelArg(ocl.clKernel, 1, sizeof(cl_mem), &d_flow_vector_array);
#endif
        clSetKernelArg(ocl.clKernel, 2, sizeof(int), &n_flow_vectors);
#ifdef OCL_2_0
        clSetKernelArgSVMPointer(ocl.clKernel, 3, d_random_numbers);
#else
        clSetKernelArg(ocl.clKernel, 3, sizeof(cl_mem), &d_random_numbers);
#endif
        clSetKernelArg(ocl.clKernel, 4, sizeof(int), &p.max_iter);
        clSetKernelArg(ocl.clKernel, 5, sizeof(int), &p.error_threshold);
        clSetKernelArg(ocl.clKernel, 6, sizeof(float), &p.convergence_threshold);
#ifdef OCL_2_0
        clSetKernelArgSVMPointer(ocl.clKernel, 7, d_g_out_id);
        clSetKernelArg(ocl.clKernel, 8, sizeof(std::atomic_int), NULL);
        clSetKernelArgSVMPointer(ocl.clKernel, 9, d_model_candidate);
        clSetKernelArgSVMPointer(ocl.clKernel, 10, d_outliers_candidate);
#else
        clSetKernelArg(ocl.clKernel, 7, sizeof(cl_mem), &d_g_out_id);
        clSetKernelArg(ocl.clKernel, 8, sizeof(int), NULL);
        clSetKernelArg(ocl.clKernel, 9, sizeof(cl_mem), &d_model_candidate);
        clSetKernelArg(ocl.clKernel, 10, sizeof(cl_mem), &d_outliers_candidate);
#endif
#ifdef FPGA
        clSetKernelArg(ocl.clKernel, 11, sizeof(cl_mem), &d_partitioner);
#else
       clSetKernelArg(ocl.clKernel, 11, sizeof(Partitioner), &partitioner);
#endif 
#ifdef OCL_2_0
        clSetKernelArg(ocl.clKernel, 12, sizeof(int), NULL);
        clSetKernelArgSVMPointer(ocl.clKernel, 13, worklist);
#endif

        // Kernel launch
        if(p.n_work_groups > 0) {
            size_t ls[1] = {(size_t)p.n_work_items};
            size_t gs[1] = {(size_t)p.n_work_groups * p.n_work_items};
            clStatus     = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
            CL_ERR();
        }
        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_model_candidate, h_outliers_candidate, h_model_param_local,
            h_flow_vector_array, n_flow_vectors, h_random_numbers, p.max_iter, p.error_threshold,
            p.convergence_threshold, h_g_out_id, p.n_threads, partitioner
#ifdef OCL_2_0
            ,
            worklist);
#else
            );
#endif

        clFinish(ocl.clCommandQueue);
        main_thread.join();

        if(rep >= p.n_warmup)
            timer.stop("Kernel");

#ifndef OCL_2_0
        // Copy back
        if(rep >= p.n_warmup)
            timer.start("Copy Back and Merge");
        int d_candidates;
        if(p.alpha < 1.0) {
            clStatus = clEnqueueReadBuffer(
                ocl.clCommandQueue, d_g_out_id, CL_TRUE, 0, sizeof(int), &d_candidates, 0, NULL, NULL);
            clStatus = clEnqueueReadBuffer(ocl.clCommandQueue, d_model_candidate, CL_TRUE, 0,
                d_candidates * sizeof(int), &h_model_candidate[h_g_out_id[0]], 0, NULL, NULL);
            clStatus = clEnqueueReadBuffer(ocl.clCommandQueue, d_outliers_candidate, CL_TRUE, 0,
                d_candidates * sizeof(int), &h_outliers_candidate[h_g_out_id[0]], 0, NULL, NULL);
            CL_ERR();
        }
        h_g_out_id[0] += d_candidates;
        clFinish(ocl.clCommandQueue);
        if(rep >= p.n_warmup)
            timer.stop("Copy Back and Merge");
#endif

        // Post-processing (chooses the best model among the candidates)
        if(rep >= p.n_warmup)
            timer.start("Kernel");
        for(int i = 0; i < h_g_out_id[0]; i++) {
            if(h_outliers_candidate[i] < best_outliers) {
                best_outliers = h_outliers_candidate[i];
                best_model    = h_model_candidate[i];
            }
        }
        if(rep >= p.n_warmup)
            timer.stop("Kernel");
    }
    timer.print("Kernel", p.n_reps);
    timer.print("Copy Back and Merge", p.n_reps);

    // Verify answer
    verify(h_flow_vector_array, n_flow_vectors, h_random_numbers, p.max_iter, p.error_threshold,
        p.convergence_threshold, h_g_out_id[0], best_outliers);

    // Free memory
    timer.start("Deallocation");
#ifdef OCL_2_0
    clSVMFree(ocl.clContext, h_model_candidate);
    clSVMFree(ocl.clContext, h_outliers_candidate);
    clSVMFree(ocl.clContext, h_model_param_local);
    clSVMFree(ocl.clContext, h_g_out_id);
    clSVMFree(ocl.clContext, h_flow_vector_array);
    clSVMFree(ocl.clContext, h_random_numbers);
    clSVMFree(ocl.clContext, worklist);
#else
    free(h_model_candidate);
    free(h_outliers_candidate);
    free(h_model_param_local);
    free(h_g_out_id);
    free(h_flow_vector_array);
    free(h_random_numbers);
    clReleaseMemObject(d_model_candidate);
    clReleaseMemObject(d_outliers_candidate);
    clReleaseMemObject(d_model_param_local);
    clReleaseMemObject(d_g_out_id);
    clReleaseMemObject(d_flow_vector_array);
    clReleaseMemObject(d_random_numbers);
#endif
    ocl.release();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    return 1;
}
