project(task_initial)

set(TASK_INITIAL_1 "task_initial_1")
set(TASK_INITIAL_2 "task_initial_2")
set(TASK_INITIAL_3 "task_initial_3")

set(TASK_INITIAL_1_SRC
    task_1.cpp
)

set(TASK_INITIAL_2_SRC
    task_2.cpp
)

set(TASK_INITIAL_3_SRC
    task_3.cpp
)

add_executable(${TASK_INITIAL_1} ${TASK_INITIAL_1_SRC})
add_executable(${TASK_INITIAL_2} ${TASK_INITIAL_2_SRC})
add_executable(${TASK_INITIAL_3} ${TASK_INITIAL_3_SRC})

