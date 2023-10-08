project(task_matrix)

find_package(benchmark REQUIRED)

file(GLOB TASK_MATRIX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.*)
add_executable(task_matrix ${TASK_MATRIX_SRC})

target_link_libraries(task_matrix benchmark::benchmark)
