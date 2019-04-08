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

#ifndef _PARTITIONER_H_
#define _PARTITIONER_H_

#include <atomic>
#include <iostream>

#define FPGA
#define STATIC_PARTITIONING 0
#define DYNAMIC_PARTITIONING 1

typedef struct Partitioner {

    unsigned int n_tasks;
    unsigned int cut;
    unsigned int strategy;

} Partitioner;

inline Partitioner partitioner_create(unsigned int n_tasks, float alpha) {
    Partitioner p;
    p.n_tasks = n_tasks;
    if(alpha >= 0.0 && alpha <= 1.0) {
        p.cut      = p.n_tasks * alpha;
        p.strategy = STATIC_PARTITIONING;
    } else {
        p.strategy = DYNAMIC_PARTITIONING;
    }
    return p;
}

inline unsigned int cpu_first(const Partitioner *p, unsigned int id, std::atomic_int *worklist) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        return worklist->fetch_add(1);
    } else {
        return (id * p->cut);
    }
}

inline unsigned int cpu_next(
    const Partitioner *p, unsigned int old, unsigned int numCPUThreads, std::atomic_int *worklist) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        return worklist->fetch_add(1);
    } else {
        return old + numCPUThreads;
    }
}

inline bool cpu_more(const Partitioner *p, unsigned int id, unsigned int old) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        return (old < p->n_tasks);
    } else {
        return (old < (id == 0 ? p->cut : p->n_tasks));
    }
}

#endif
