#########################################################################

# CMAKE MINIMAL VERSION
cmake_minimum_required (VERSION 3.13)

#########################################################################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILER
    /usr/bin/mpic++
)

# COMPILE OPTIONS
add_compile_options(
    -mavx
    -fopenmp
    -Wall
    -Wextra
    -Werror
)

# LINK OPTIONS
add_link_options(
    -fopenmp
)

#########################################################################

# PROJECT PROPERTIES
project ("Parallel programming" LANGUAGES CXX)


set(DEBUG_COMPILER_OPTIONS
    -g -O1 -DDBG
)

set(RELEASE_COMPILER_OPTIONS
    -O2
)

#########################################################################

include_directories(
    include/
)

#########################################################################

add_subdirectory(tasks/lab_1)
add_subdirectory(tasks/lab_2)
add_subdirectory(tasks/task_copyin)
add_subdirectory(tasks/task_exponent)
add_subdirectory(tasks/task_initial)
add_subdirectory(tasks/task_initial_omp)
add_subdirectory(tasks/task_matrix)
add_subdirectory(tasks/task_nested)
add_subdirectory(tasks/task_new_comm)
add_subdirectory(tasks/task_pthread)
add_subdirectory(tasks/task_schedule)
add_subdirectory(tasks/task_send)
add_subdirectory(tasks/task_sort)
add_subdirectory(tasks/task_sort_omp)

#########################################################################


foreach(TARGET IN LISTS TARGETS)

    option(DEBUG_BUILD "Build debug project version." OFF)

    if (DEBUG_BUILD)
        target_compile_options(${TARGET} PRIVATE ${DEBUG_COMPILER_OPTIONS})
    else()
        target_compile_options(${TARGET} PRIVATE ${RELEASE_COMPILER_OPTIONS})
    endif (DEBUG_BUILD)

endforeach()


