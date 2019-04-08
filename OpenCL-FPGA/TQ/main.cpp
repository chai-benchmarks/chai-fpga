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
#include "support/timer.h"
#include "support/verify.h"

#include <string.h>
#include <unistd.h>
#include <thread>
#include <vector>

struct Params {

    int         platform;
    int         device;
    int         n_work_items;
    int         n_work_groups;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    const char *file_name;
    int         pattern;
    int         pool_size;
    int         queue_size;
    int         iterations;

    Params(int argc, char **argv) {
        platform      = 0;
        device        = 0;
        n_work_items  = 64;
        n_work_groups = 8 * 40;
        n_threads     = 1;
        n_warmup      = 5;
        n_reps        = 50;
        file_name     = "input/patternsNP100NB512FB25.txt";
        pattern       = 1;
        pool_size     = 3200;
        queue_size    = 320;
        iterations    = 50;
        char opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:f:k:s:q:n:")) >= 0) {
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
            case 'f': file_name     = optarg; break;
            case 'k': pattern       = atoi(optarg); break;
            case 's': pool_size     = atoi(optarg); break;
            case 'q': queue_size    = atoi(optarg); break;
            case 'n': iterations    = atoi(optarg); break;
            default:
                cerr << "\nUnrecognized option!" << endl;
                usage();
                exit(0);
            }
        }
    }

    void usage() {
        cerr << "\nUsage:  ./tq [options]"
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
                "\nBenchmark-specific options:"
                "\n    -f <F>    patterns file name"
                "\n    -k <K>    pattern in file"
                "\n    -s <S>    task pool size"
                "\n    -q <Q>    task queue size"
                "\n    -n <N>    # of iterations in heavy task"
                "\n";
    }
};

// Input Data -----------------------------------------------------------------
vector<string> tokenize(const string &str, const string &delimiters) {
    vector<string> tokens;
    // skip delimiters at beginning.
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // find first "non-delimiter".
    string::size_type pos = str.find_first_of(delimiters, lastPos);
    while(string::npos != pos || string::npos != lastPos) {
        // found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
    return tokens;
};

void read_input(int *pattern, task_t *task_pool, const Params &p) {

    // Patterns file name
    char filePatterns[100];

    sprintf(filePatterns, "%s", p.file_name);

    // Read line from patterns file
    ifstream myfile;
    string   line;
    myfile.open(filePatterns);
    if(!myfile.is_open()) {
        printf("%s does not exist\n", filePatterns);
        exit(1);
    }

    for(int x = 0; x <= p.pattern; x++) {
        getline(myfile, line);
    }
    myfile.close();

    //Create a random index from 0 to NBLOCKS - 1 for the computation block
    for(int i = 0; i < p.pool_size; i++) {
        pattern[i] = 0;
        //Setting tasks in the tasks pool
        task_pool[i].id = i;
        task_pool[i].op = SIGNAL_NOTWORK_KERNEL;
    }

    //Read the pattern
    vector<string> v(tokenize(line, " "));
    for(int i = 0; i < p.pool_size; i++) {
        //int c = atoi(v[i].c_str());
        int c      = atoi(v[i % 512].c_str());
        pattern[i] = c;
        if(pattern[i] == 1) {
            task_pool[i].op = SIGNAL_WORK_KERNEL;
        }
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
    int *   h_pattern     = (int *)alignedMalloc(p.pool_size * sizeof(int));
    task_t *h_task_pool   = (task_t *)alignedMalloc(p.pool_size * sizeof(task_t));
    task_t *h_task_queues = (task_t *)alignedMalloc(p.queue_size * sizeof(task_t));
    cl_mem  d_task_queues = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, p.queue_size * sizeof(task_t), NULL, 0);
    int *   h_data_pool   = (int *)alignedMalloc(p.pool_size * p.n_work_items * sizeof(int));
    int *   h_data_queues = (int *)alignedMalloc(p.queue_size * p.n_work_items * sizeof(int));
    cl_mem  d_data_queues =
        clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, p.queue_size * p.n_work_items * sizeof(int), NULL, 0);
    int *  h_consumed = (int *)alignedMalloc(sizeof(int));
    cl_mem d_consumed = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(int), NULL, 0);
    clFinish(ocl.clCommandQueue);
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    read_input(h_pattern, h_task_pool, p);
    memset((void *)h_data_pool, 0, p.pool_size * p.n_work_items * sizeof(int));
    memset((void *)h_consumed, 0, sizeof(int));
    timer.stop("Initialization");
    timer.print("Initialization", 1);

    for(int rep = 0; rep < p.n_reps + p.n_warmup; rep++) {

        // Reset
        memset((void *)h_data_pool, 0, p.pool_size * p.n_work_items * sizeof(int));
        int n_written_tasks = 0;

        for(int n_consumed_tasks = 0; n_consumed_tasks < p.pool_size; n_consumed_tasks += p.queue_size) {

            if(rep >= p.n_warmup)
                timer.start("Kernel");
            host_insert_tasks(h_task_queues, h_data_queues, h_task_pool, h_data_pool, &n_written_tasks, p.queue_size,
                n_consumed_tasks, p.n_work_items);
            if(rep >= p.n_warmup)
                timer.stop("Kernel");

            if(rep >= p.n_warmup)
                timer.start("Copy To Device");
            clEnqueueWriteBuffer(ocl.clCommandQueue, d_task_queues, CL_TRUE, 0, p.queue_size * sizeof(task_t),
                h_task_queues, 0, NULL, NULL);
            clEnqueueWriteBuffer(ocl.clCommandQueue, d_data_queues, CL_TRUE, 0,
                p.queue_size * p.n_work_items * sizeof(int), h_data_queues, 0, NULL, NULL);
            clEnqueueWriteBuffer(ocl.clCommandQueue, d_consumed, CL_TRUE, 0, sizeof(int), h_consumed, 0, NULL, NULL);
            clFinish(ocl.clCommandQueue);
            if(rep >= p.n_warmup)
                timer.stop("Copy To Device");

            if(rep >= p.n_warmup)
                timer.start("Kernel");
            // Setting kernel arguments
            clSetKernelArg(ocl.clKernel, 0, sizeof(task_t *), &d_task_queues);
            clSetKernelArg(ocl.clKernel, 1, sizeof(int *), &d_data_queues);
            clSetKernelArg(ocl.clKernel, 2, sizeof(cl_mem), &d_consumed);
            clSetKernelArg(ocl.clKernel, 3, sizeof(int), &p.iterations);
            clSetKernelArg(ocl.clKernel, 4, sizeof(int), &n_consumed_tasks);
            clSetKernelArg(ocl.clKernel, 5, sizeof(int), &p.queue_size);
            clSetKernelArg(ocl.clKernel, 6, sizeof(task_t), NULL);
            clSetKernelArg(ocl.clKernel, 7, sizeof(int), NULL);
            // Kernel launch
            size_t ls[1] = {(size_t)p.n_work_items};
            size_t gs[1] = {(size_t)p.n_work_groups * p.n_work_items};
            clStatus     = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
            CL_ERR();
            clFinish(ocl.clCommandQueue);
            if(rep >= p.n_warmup)
                timer.stop("Kernel");

            if(rep >= p.n_warmup)
                timer.start("Copy Back and Merge");
            clEnqueueReadBuffer(ocl.clCommandQueue, d_data_queues, CL_TRUE, 0,
                p.queue_size * p.n_work_items * sizeof(int), &h_data_pool[n_consumed_tasks * p.n_work_items], 0, NULL,
                NULL);
            clFinish(ocl.clCommandQueue);
            if(rep >= p.n_warmup)
                timer.stop("Copy Back and Merge");
        }
    }
    timer.print("Copy To Device", p.n_reps);
    timer.print("Kernel", p.n_reps);
    timer.print("Copy Back and Merge", p.n_reps);

    // Verify answer
    verify(h_data_pool, h_pattern, p.pool_size, p.iterations, p.n_work_items);

    // Free memory
    timer.start("Deallocation");
    clReleaseMemObject(d_task_queues);
    clReleaseMemObject(d_data_queues);
    clReleaseMemObject(d_consumed);
    free(h_consumed);
    free(h_task_queues);
    free(h_data_queues);
    free(h_task_pool);
    free(h_data_pool);
    ocl.release();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    return 1;
}
