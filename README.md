Collaborative Computing on Heterogeneous CPU-FPGA Architectures Using OpenCL
=======

## Software and Hardware requirements
This repository contains the synthesizable OpenCL code of [Chai benchmarks](https://chai-benchmarks.github.io/). This work studies the OpenCL synthesis for FPGAs using Intel OpenCL SDK for FPGA. Therefore, the evaluation of the work requires Intel Quartus Prime software (including OpenCL SDK for FPGA), its license, and FPGA hardware. The FPGA synthesis software and FPGA hardware used in the paper are listed below: 
  * FPGA synthesis tool: Intel Quartus Prime Pro 16.0
  * FPGA board support package (BSP) provided by [Terasic](https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&CategoryNo=231&No=970&PartNo=4)
  * FPGA device: Terasic DE5a-Net
  * Host compiler: GCC 7.4.0
  * Operating system: Ubuntu 16.04

## Install Intel Quartus Prime and OpenCL SDK
Please refer to Intel's OpenCL guide for details of installing Intel OpenCL SDK. The guideline of setting up Intel OpenCL SDK can be found [here][intel_opencl]. 

[intel_opencl]: https://www.intel.com/content/dam/www/programmable/us/en/pdfs/literature/hb/opencl-sdk/archives/ug-aocl-getting-started-16.0.pdf


## Install FPGA Board and Driver
Please refer to vendor's manual for detailed steps of installing FPGA board and driver. The step of installation can be different for different FPGA board and operating system. For Terasic DE5a-Net board, please refer to [DE5a-Net OpenCL Manual][a10_opencl]. Note that there are many environment variables need to setup properly. 

[a10_opencl]: http://download.terasic.com/downloads/cd-rom/de5a-net/linux_BSP/I2/DE5ANET_I2_OpenCL_16.1.pdf

To install the driver: 
```bash
    aocl install
```


## FPGA Design Synthesis
Before running FPGA design synthesis, please make sure Intel Quartus Prime Pro 16.0 is properly installed and license are available. 

```bash
    cd OpenCL-FPGA/BS
    aoc -v BS.cl -o bin_a10_BS/BS.aocx
```
This will take several hours to finish. After synthesis is done, copy the generated `aocx` file back to the benchmark directory: 
```bash
    cp bin_a10_BS/BS.aocx ..
```

## Compile host code
Under the directory of each benchmark, there is one `Makefile`. To compile OpenCL host code, simply navigate to the benchmark directory and `make`. For example, to compile the OpenCL host code for benchmark `BS`:
```bash
    cd OpenCL-FPGA/BS
    make
```
An executable file `bs` will be generated. To clean the make: 
```bash
    make clean
```

## Run

To run a specific benchmark, simply `cd` to the benchmark and execute the host executable. For example, to execute benchmark `BS`:
```bash
    cd OpenCL-FPGA/BS
    ./bs
```

The example output is: 

```
    a10gx : Arria 10 Reference Platform (acla10_ref0)       Using AOCX: BS.aocx
    Reprogramming device with handle 1
    Allocation Time (ms): 0.027
    Initialization Time (ms): 11.763
    Copy To Device Time (ms): 0.029
    Kernel Time (ms): 74.3389
    Copy Back and Merge Time (ms): 0.908
    TEST PASSED
    Deallocation Time (ms): 6.451
```

# Reference
You can refer to our paper for further details. 

Sitao Huang, Li-Wen Chang, Izzat El Hajj, Simon Garcia de Gonzalo, Juan Gómez Luna, Sai Rahul Chalamalasetti, Mohamed El-Hadedy, Dejan Milojicic, Onur Mutlu, Deming Chen, and Wen-mei Hwu. Analysis and Modeling of Collaborative Execution Strategies for Heterogeneous CPU-FPGA Architectures. Proceedings of *10th ACM/SPEC International Conference on Performance Engineering (**ICPE 2019**)*, 2019. 



## Contact
If you have any questions, feel free to contact Sitao Huang <shuang91@illinois.edu>. 

