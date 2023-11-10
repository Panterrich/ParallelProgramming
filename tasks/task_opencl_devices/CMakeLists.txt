project(task_opencl_devices)

file(GLOB TASK_OPENCL_DEVICES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

add_executable(task_opencl_devices ${TASK_OPENCL_DEVICES_SRC})
