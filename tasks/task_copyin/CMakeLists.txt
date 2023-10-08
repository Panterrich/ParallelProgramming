project(task_copyin)

file(GLOB TASK_COPYIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

add_executable(task_copyin ${TASK_COPYIN_SRC})
