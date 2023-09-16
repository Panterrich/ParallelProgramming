project(task_schedule)

file(GLOB TASK_SCHEDULE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

add_executable(task_schedule ${TASK_SCHEDULE_SRC})
