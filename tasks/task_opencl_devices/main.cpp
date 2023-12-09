#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <stdio.h>
#include <iostream>
#include <CL/opencl.hpp>

int main()
{
    // get all platforms (drivers), e.g. NVIDIA
    std::vector<cl::Platform> allPlatforms;
    cl::Platform::get(&allPlatforms);

    if (allPlatforms.empty())
    {
        std::cout << "No platforms found. Check OpenCL installation!\n";
        return 1;
    }

    for (auto it: allPlatforms)
    {
        std::cout << "Platform: " << it.getInfo<CL_PLATFORM_NAME>() << std::endl;
    }

    cl::Platform defaultPlatform = allPlatforms[0];

    std::cout << "Using platform: " << defaultPlatform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    // get default device (CPUs, GPUs) of the default platform
    std::vector<cl::Device> allDevices;
    defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);

    if (allDevices.empty())
    {
        std::cout << "No devices found. Check OpenCL installation!\n";
        return 1;
    }

    for (auto it: allDevices)
    {
        std::cout << "Device: " << it.getInfo<CL_DEVICE_NAME>() << std::endl;
    }

    cl::Device defaultDevice = allDevices[0];
    std::cout << "Using device: " << defaultDevice.getInfo<CL_DEVICE_NAME>() << std::endl;


    // CL_DEVICE_MAX_COMPUTE_UNITS        - A compute unit is the device component that runs a work-group,
    //                                      i.e. a (cooperating) collection of work-items.
    //                                      Each work-group in an NDRange is assigned to one (and only one) compute unit,
    //                                      although a compute unit may (be able to) run multiple work-groups at once.
    // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS - Maximum dimensions that specify the global and
    //                                      local work-item IDs used by the data parallel execution model.
    // CL_DEVICE_MAX_WORK_GROUP_SIZE      - Maximum number of work-items in a work-group
    // CL_DEVICE_MAX_WORK_ITEM_SIZES      - Maximum number of work-items that can be specified in each dimension of the work-group
    // CL_DEVICE_MAX_CLOCK_FREQUANCY      - Maximum configured clock frequency of the device in MHz.
    // CL_DEVICE_ADDRESS_BITS             - The default compute device address space size of
    //                                      the global address space specified as an unsigned integer value in bits
    // CL_DEVICE_IMAGE_SUPPORT            -

    std::cout << "\n"
                 "Features:\n"
              << "Device type: "              << defaultDevice.getInfo<CL_DEVICE_TYPE>()                      << std::endl
              << "Max compute units: "        << defaultDevice.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()         << std::endl
              << "Max work item dimensions: " << defaultDevice.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>()  << std::endl
              << "Max work group size: "      << defaultDevice.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()       << std::endl
              << "Max work item sizes: "      << defaultDevice.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0]    << std::endl
              << "Max clock frequancy: "      << defaultDevice.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()       << std::endl
              << "Address bits: "             << defaultDevice.getInfo<CL_DEVICE_ADDRESS_BITS>()              << std::endl
              << "Image support: "            << defaultDevice.getInfo<CL_DEVICE_IMAGE_SUPPORT>()             << std::endl
              << "Cache type: "               << defaultDevice.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>()     << std::endl
              << "Cacheline size: "           << defaultDevice.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>() << std::endl
              << "Cache size: "               << defaultDevice.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>()     << std::endl
              << "Global mem size: "          << defaultDevice.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()           << std::endl
              << "Local mem type: "           << defaultDevice.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>()            << std::endl
              << "Local mem size: "           << defaultDevice.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()            << std::endl;
}


