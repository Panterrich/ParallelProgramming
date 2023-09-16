#########################################################################

# CMAKE MINIMAL VERSION
cmake_minimum_required (VERSION 3.13)

#########################################################################

find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILER
    /usr/bin/mpic++
)

# COMPILE OPTIONS
add_compile_options(
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

add_subdirectory(lab_1)
add_subdirectory(lab_2)
add_subdirectory(task_copyin)
add_subdirectory(task_exponent)
add_subdirectory(task_initial)
add_subdirectory(task_initial_omp)
add_subdirectory(task_nested)
add_subdirectory(task_new_comm)
add_subdirectory(task_pthread)
add_subdirectory(task_schedule)
add_subdirectory(task_send)
add_subdirectory(task_sort)

#########################################################################


foreach(TARGET IN LISTS TARGETS)

    option(DEBUG_BUILD "Build debug project version." ON)

    if (DEBUG_BUILD)
        target_compile_options(${TARGET} PRIVATE ${DEBUG_COMPILER_OPTIONS})
    else()
        target_compile_options(${TARGET} PRIVATE ${RELEASE_COMPILER_OPTIONS})
    endif (DEBUG_BUILD)

endforeach()


