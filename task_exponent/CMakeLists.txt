project(task_exponent)

file(GLOB TASK_EXPONENT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

add_executable(task_exponent ${TASK_EXPONENT_SRC})
target_link_libraries(task_exponent gmp)
