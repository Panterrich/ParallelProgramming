project(task_sort_omp)

find_package(benchmark REQUIRED)

file(GLOB TASK_SORT_OMP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.*)
add_executable(task_sort_omp ${TASK_SORT_OMP_SRC})

target_link_libraries(task_sort_omp benchmark::benchmark)
