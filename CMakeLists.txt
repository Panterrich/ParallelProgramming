#########################################################################

# CMAKE MINIMAL VERSION
cmake_minimum_required (VERSION 3.11)

# PROJECT PROPERTIES
project ("Parallel programming" LANGUAGES CXX)

#########################################################################

find_package(MPI REQUIRED)

# COMPILE OPTIONS
add_compile_options(
    -Wall

    # -Wno-pointer-to-int-cast
    # -Wno-int-to-pointer-cast
    # -Wno-variadic-macros
    # -Wno-unused-variable
    # -Wno-unused-parameter
    # -Wno-unused-function
    # -Wno-sign-compare
    # -Wno-misleading-indentation
    # -Wno-format
)

set(CMAKE_CXX_COMPILER
    /usr/bin/mpic++
)

set(DEBUG_COMPILER_OPTIONS
    -g -O1 -DDBG
)

set(RELEASE_COMPILER_OPTIONS
    -O2
)

#########################################################################

# EXECUTABLE FILES PROPERTIES
set(TASK_INITIAL_1 "task_initial_1")
set(TASK_INITIAL_2 "task_initial_2")
set(TASK_INITIAL_3 "task_initial_3")
set(TASK_EXPONENT  "task_exponent")
set(TASK_NEW_COMM  "task_new_comm")

set(TASK_INITIAL_1_SRC
    task_initial/task_1.cpp
)

set(TASK_INITIAL_2_SRC
    task_initial/task_2.cpp
)   

set(TASK_INITIAL_3_SRC
    task_initial/task_3.cpp
)

set(TASK_EXPONENT_SRC
    task_exponent/task_exponent.cpp
)

set(TASK_NEW_COMM_SRC
    task_new_comm/task_new_comm.cpp
)

# EXECUTABLE FILES
add_executable(${TASK_INITIAL_1} ${TASK_INITIAL_1_SRC})
add_executable(${TASK_INITIAL_2} ${TASK_INITIAL_2_SRC})
add_executable(${TASK_INITIAL_3} ${TASK_INITIAL_3_SRC})
add_executable(${TASK_EXPONENT}  ${TASK_EXPONENT_SRC})
add_executable(${TASK_NEW_COMM}  ${TASK_NEW_COMM_SRC})

target_link_libraries(${TASK_EXPONENT} gmp)

target_include_directories(${TASK_INITIAL_1} PRIVATE include/)
target_include_directories(${TASK_INITIAL_2} PRIVATE include/)
target_include_directories(${TASK_INITIAL_3} PRIVATE include/)
target_include_directories(${TASK_EXPONENT}  PRIVATE include/)
target_include_directories(${TASK_NEW_COMM}  PRIVATE include/)

#########################################################################

foreach(TARGET IN LISTS TARGETS)

    option(DEBUG_BUILD "Build debug project version." ON)

    if (DEBUG_BUILD)
        target_compile_options(${TARGET} PRIVATE ${DEBUG_COMPILER_OPTIONS})
    else()
        target_compile_options(${TARGET} PRIVATE ${RELEASE_COMPILER_OPTIONS})
    endif (DEBUG_BUILD)

endforeach()

#########################################################################
