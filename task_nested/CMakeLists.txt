project(task_nested)

file(GLOB TASK_NESTED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

add_executable(task_nested ${TASK_NESTED_SRC})
