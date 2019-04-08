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

#include <stdio.h>
#include <unistd.h>
#include <thread>
#include <vector>
#include <cstring>
#include <algorithm>

// Params ---------------------------------------------------------------------
struct Params {

    int         platform;
    int         device;
    int         n_work_items;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    float       alpha;
    const char *file_name;
    const char *comparison_file;

    Params(int argc, char **argv) {
        platform        = 0;
        device          = 0;
        n_work_items    = 16;
        n_threads       = 4;
        n_warmup        = 10;
        n_reps          = 100;
        alpha           = 0.2;
        file_name       = "input/PeppaPigandSuzieSheepWhistle.raw";
        comparison_file = "output/Peppa.txt";
        char opt;
        while((opt = getopt(argc, argv, "hp:d:i:t:w:r:a:f:c:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'p': platform        = atoi(optarg); break;
            case 'd': device          = atoi(optarg); break;
            case 'i': n_work_items    = atoi(optarg); break;
            case 't': n_threads       = atoi(optarg); break;
            case 'w': n_warmup        = atoi(optarg); break;
            case 'r': n_reps          = atoi(optarg); break;
            case 'a': alpha           = atof(optarg); break;
            case 'f': file_name       = optarg; break;
            case 'c': comparison_file = optarg; break;
            default:
                cerr << "\nUnrecognized option!" << endl;
                usage();
                exit(0);
            }
        }
    }

    void usage() {
        cerr << "\nUsage:  ./cedd [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -p <P>    OpenCL platform ID (default=0)"
                "\n    -d <D>    OpenCL device ID (default=0)"
                "\n    -i <I>    # of device work-items"
                "\n    -t <T>    # of host threads"
                "\n    -w <W>    # of untimed warmup iterations"
                "\n    -r <R>    # of timed repition iterations"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of input elements to process on host"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    input video file name"
                "\n    -c <C>    comparison file"
                "\n";
    }
};

// Input Data -----------------------------------------------------------------
void read_input(unsigned char **all_gray_frames, int &rowsc, int &colsc, int &in_size, const Params &p) {

    FILE* pFile = fopen (p.file_name,"r");
    int framec=0;
    fscanf(pFile, "%d %d %d\n", &framec, &rowsc, &colsc);

    in_size    = rowsc * colsc * sizeof(unsigned char);

    for(int k=0;k<framec;k++) {
        all_gray_frames[k] = (unsigned char*) alignedMalloc(in_size);
        for(int i = 0; i < rowsc; i++) {
            for(int j = 0; j < colsc; j++) {
                unsigned int temp;
                fscanf(pFile,"%u ",&temp);
                all_gray_frames[k][i * colsc + j] = (unsigned char)temp;
            }
        }
    }
    fclose(pFile);
    
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    Params      p(argc, argv);
    OpenCLSetup ocl(p.platform, p.device);
    cl_int      clStatus;
    Timer       timer;

    // Initialize (part 1)
    timer.start("Initialization");
    unsigned char* all_gray_frames[p.n_warmup + p.n_reps];
    int     rowsc, colsc, in_size;
    read_input(all_gray_frames, rowsc, colsc, in_size, p);
    timer.stop("Initialization");

    // Allocate buffers
    timer.start("Allocation");
    const unsigned int CPU_PROXY = 0;
    const unsigned int FPGA_PROXY = 1;
    unsigned char *    h_in_out[2];
    h_in_out[CPU_PROXY] = (unsigned char *)alignedMalloc(in_size);
#ifdef OCL_2_0
    h_in_out[FPGA_PROXY]     = (unsigned char *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, in_size, 0);
    unsigned char *d_in_out = h_in_out[FPGA_PROXY];
#else
    h_in_out[FPGA_PROXY] = (unsigned char *)alignedMalloc(in_size);
    cl_mem d_in_out     = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, in_size, NULL, &clStatus);
#endif
    unsigned char *h_interm_cpu_proxy = (unsigned char *)alignedMalloc(in_size);
    unsigned char *h_theta_cpu_proxy  = (unsigned char *)alignedMalloc(in_size);
    cl_mem         d_interm_gpu_proxy = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, in_size, NULL, &clStatus);
    cl_mem         d_theta_gpu_proxy  = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, in_size, NULL, &clStatus);
    // TODO: why not allocate filters in SVM for OpenCL 2.0
    float            h_gaus[3][3] = {{0.0625, 0.125, 0.0625}, {0.1250, 0.250, 0.1250}, {0.0625, 0.125, 0.0625}};
    int              h_sobx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int              h_soby[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    cl_mem           d_gaus = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, 3 * 3 * sizeof(float), NULL, &clStatus);
    cl_mem           d_sobx = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, 3 * 3 * sizeof(int), NULL, &clStatus);
    cl_mem           d_soby = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, 3 * 3 * sizeof(int), NULL, &clStatus);
    std::atomic<int> next_frame;
    clFinish(ocl.clCommandQueue);
    CL_ERR();
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize (part 2)
    timer.start("Initialization");
    unsigned char* all_out_frames[p.n_warmup + p.n_reps];
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        all_out_frames[i] = (unsigned char*) alignedMalloc(in_size);
        memcpy(all_out_frames[i], all_gray_frames[i], in_size);
    }
    Partitioner      partitioner = partitioner_create(p.n_warmup + p.n_reps, p.alpha);
    std::atomic_int *worklist    = (std::atomic_int *)alignedMalloc(sizeof(std::atomic_int));
    if(partitioner.strategy == DYNAMIC_PARTITIONING) {
        worklist[0].store(0);
    }
    next_frame.store(0);
    timer.stop("Initialization");
    timer.print("Initialization", 1);

    // Copy to device
    // TODO: why not allocate filters in SVM for OpenCL 2.0
    timer.start("Copy To Device");
    clStatus =
        clEnqueueWriteBuffer(ocl.clCommandQueue, d_gaus, CL_TRUE, 0, 3 * 3 * sizeof(float), h_gaus, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_sobx, CL_TRUE, 0, 3 * 3 * sizeof(int), h_sobx, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_soby, CL_TRUE, 0, 3 * 3 * sizeof(int), h_soby, 0, NULL, NULL);
    timer.stop("Copy To Device");
    timer.print("Copy To Device", 1);

    timer.start("Total Proxies");
    std::vector<std::thread> proxy_threads;
    for(int proxy_tid = 0; proxy_tid < 2; proxy_tid++) {
        proxy_threads.push_back(std::thread([&, proxy_tid]() {

            for(int task_id = cpu_first(&partitioner, proxy_tid, worklist); cpu_more(&partitioner, proxy_tid, task_id);
                task_id     = cpu_next(&partitioner, task_id, 1 /*see note*/,
                    worklist)) { // note: it must be 1 because there is one single proxy thread fetching frames for each device

                unsigned char* gray_frame;

                // Next frame
                gray_frame = all_gray_frames[task_id];
                if(gray_frame==0)
                    continue;
                memcpy(h_in_out[proxy_tid], gray_frame, in_size);

                if(proxy_tid == FPGA_PROXY) {

#ifndef OCL_2_0
                    // Copy to Device
                    timer.start("FPGA Proxy: Copy To Device");
                    clStatus = clEnqueueWriteBuffer(
                        ocl.clCommandQueue, d_in_out, CL_TRUE, 0, in_size, h_in_out[proxy_tid], 0, NULL, NULL);
                    CL_ERR();
                    clFinish(ocl.clCommandQueue);
                    timer.stop("FPGA Proxy: Copy To Device");
#endif

                    timer.start("FPGA Proxy: Kernel");
                    // Execution configuration
                    size_t ls[2]     = {(size_t)p.n_work_items, (size_t)p.n_work_items};
#ifdef FPGA
                    size_t gs[2]     = {(size_t)(rowsc), (size_t)(colsc)};
                    size_t *offset   = NULL;
#else
                    size_t gs[2]     = {(size_t)(rowsc - 2), (size_t)(colsc - 2)};
                    size_t offset[2] = {(size_t)1, (size_t)1};
#endif
// GAUSSIAN KERNEL
// Set arguments
#ifdef OCL_2_0
                    clSetKernelArgSVMPointer(ocl.clKernel_gauss, 0, d_in_out);
#else
                    clSetKernelArg(ocl.clKernel_gauss, 0, sizeof(cl_mem), &d_in_out);
#endif
                    clSetKernelArg(ocl.clKernel_gauss, 1, sizeof(cl_mem), &d_interm_gpu_proxy);
                    clSetKernelArg(ocl.clKernel_gauss, 2, sizeof(int), &rowsc);
                    clSetKernelArg(ocl.clKernel_gauss, 3, sizeof(int), &colsc);
                    clSetKernelArg(ocl.clKernel_gauss, 4, (L_SIZE + 2) * (L_SIZE + 2) * sizeof(int), NULL);
                    clSetKernelArg(ocl.clKernel_gauss, 5, sizeof(cl_mem), &d_gaus);
//		    std::cout << gs[0] << ", " << gs[1] << "; " << ls[0] << ", " << ls[1] << std::endl;
                    // Kernel launch
                    clStatus = clEnqueueNDRangeKernel(
                        ocl.clCommandQueue, ocl.clKernel_gauss, 2, offset, gs, ls, 0, NULL, NULL);
                    CL_ERR();

                    // SOBEL KERNEL
                    // Set arguments
                    clSetKernelArg(ocl.clKernel_sobel, 0, sizeof(cl_mem), &d_interm_gpu_proxy);
#ifdef OCL_2_0
                    clSetKernelArgSVMPointer(ocl.clKernel_sobel, 1, d_in_out);
#else
                    clSetKernelArg(ocl.clKernel_sobel, 1, sizeof(cl_mem), &d_in_out);
#endif
                    clSetKernelArg(ocl.clKernel_sobel, 2, sizeof(cl_mem), &d_theta_gpu_proxy);
                    clSetKernelArg(ocl.clKernel_sobel, 3, sizeof(int), &rowsc);
                    clSetKernelArg(ocl.clKernel_sobel, 4, sizeof(int), &colsc);
                    clSetKernelArg(ocl.clKernel_sobel, 5, (L_SIZE + 2) * (L_SIZE + 2) * sizeof(int), NULL);
                    clSetKernelArg(ocl.clKernel_sobel, 6, sizeof(cl_mem), &d_sobx);
                    clSetKernelArg(ocl.clKernel_sobel, 7, sizeof(cl_mem), &d_soby);
                    // Kernel launch
                    clStatus = clEnqueueNDRangeKernel(
                        ocl.clCommandQueue, ocl.clKernel_sobel, 2, offset, gs, ls, 0, NULL, NULL);
                    CL_ERR();

// NON-MAXIMUM SUPPRESSION KERNEL
// Set arguments
#ifdef OCL_2_0
                    clSetKernelArgSVMPointer(ocl.clKernel_nonmax, 0, d_in_out);
#else
                    clSetKernelArg(ocl.clKernel_nonmax, 0, sizeof(cl_mem), &d_in_out);
#endif
                    clSetKernelArg(ocl.clKernel_nonmax, 1, sizeof(cl_mem), &d_interm_gpu_proxy);
                    clSetKernelArg(ocl.clKernel_nonmax, 2, sizeof(cl_mem), &d_theta_gpu_proxy);
                    clSetKernelArg(ocl.clKernel_nonmax, 3, sizeof(int), &rowsc);
                    clSetKernelArg(ocl.clKernel_nonmax, 4, sizeof(int), &colsc);
                    clSetKernelArg(ocl.clKernel_nonmax, 5, (L_SIZE + 2) * (L_SIZE + 2) * sizeof(int), NULL);
                    // Kernel launch
                    clStatus = clEnqueueNDRangeKernel(
                        ocl.clCommandQueue, ocl.clKernel_nonmax, 2, offset, gs, ls, 0, NULL, NULL);
                    CL_ERR();
                    //clFinish(ocl.clCommandQueue);

                    // HYSTERESIS KERNEL
                    // Set arguments
                    clSetKernelArg(ocl.clKernel_hyst, 0, sizeof(cl_mem), &d_interm_gpu_proxy);
#ifdef OCL_2_0
                    clSetKernelArgSVMPointer(ocl.clKernel_hyst, 1, d_in_out);
#else
                    clSetKernelArg(ocl.clKernel_hyst, 1, sizeof(cl_mem), &d_in_out);
#endif
                    clSetKernelArg(ocl.clKernel_hyst, 2, sizeof(int), &rowsc);
                    clSetKernelArg(ocl.clKernel_hyst, 3, sizeof(int), &colsc);
                    // Kernel launch
                    clStatus =
                        clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel_hyst, 2, offset, gs, ls, 0, NULL, NULL);
                    CL_ERR();

                    clFinish(ocl.clCommandQueue);
                    timer.stop("FPGA Proxy: Kernel");

#ifndef OCL_2_0
                    timer.start("FPGA Proxy: Copy Back");
                    clStatus = clEnqueueReadBuffer(
                        ocl.clCommandQueue, d_in_out, CL_TRUE, 0, in_size, h_in_out[proxy_tid], 0, NULL, NULL);
                    clFinish(ocl.clCommandQueue);
                    timer.stop("FPGA Proxy: Copy Back");
#endif

                } else if(proxy_tid == CPU_PROXY) {
                    // Launch CPU threads
                    timer.start("CPU Proxy: Kernel");
                    std::thread main_thread(run_cpu_threads, h_in_out[proxy_tid], h_interm_cpu_proxy, h_theta_cpu_proxy,
                        rowsc, colsc, p.n_threads, task_id);
                    main_thread.join();
                    timer.stop("CPU Proxy: Kernel");
                }
;
                memcpy(all_out_frames[task_id], h_in_out[proxy_tid], in_size);
           
            }
        }));
    }
    std::for_each(proxy_threads.begin(), proxy_threads.end(), [](std::thread &t) { t.join(); });
    clFinish(ocl.clCommandQueue);
    timer.stop("Total Proxies");
    timer.print("Total Proxies", 1);
    cout << "CPU Proxy:" << endl;
    cout << "\t";
    timer.print("CPU Proxy: Kernel", 1);
    cout << "FPGA Proxy:" << endl;
    cout << "\t";
    timer.print("FPGA Proxy: Copy To Device", 1);
    cout << "\t";
    timer.print("FPGA Proxy: Kernel", 1);
    cout << "\t";
    timer.print("FPGA Proxy: Copy Back", 1);

// Display the result
#if DISPLAY // Don't remove!!!
    //got removed
#endif

    // Verify answer
    verify(all_out_frames, in_size, p.comparison_file, p.n_warmup + p.n_reps);

    // Release buffers
    timer.start("Deallocation");
#ifdef OCL_2_0
    clSVMFree(ocl.clContext, h_in_out[FPGA_PROXY]);
#else
    free(h_in_out[FPGA_PROXY]);
    clReleaseMemObject(d_in_out);
#endif
    free(h_in_out[CPU_PROXY]);
    free(h_interm_cpu_proxy);
    free(h_theta_cpu_proxy);
    clReleaseMemObject(d_interm_gpu_proxy);
    clReleaseMemObject(d_theta_gpu_proxy);
    clReleaseMemObject(d_gaus);
    clReleaseMemObject(d_sobx);
    clReleaseMemObject(d_soby);
    free(worklist);
    ocl.release();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    return 1;
}
