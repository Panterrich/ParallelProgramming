project(lab_1)

file(GLOB LAB_1_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

add_executable(lab_1 ${LAB_1_SRC})
