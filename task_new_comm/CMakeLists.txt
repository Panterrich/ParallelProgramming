project(task_new_comm)

file(GLOB TASK_NEW_COMM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

add_executable(task_new_comm ${TASK_NEW_COMM_SRC})
