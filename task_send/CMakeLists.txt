project(task_send)

file(GLOB TASK_SEND_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

add_executable(task_send ${TASK_SEND_SRC})
