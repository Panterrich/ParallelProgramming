project(task_sort)

file(GLOB TASK_SORT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

add_executable(task_sort ${TASK_SORT_SRC})
