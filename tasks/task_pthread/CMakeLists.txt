project(task_pthread)

set(TASK_PTHREAD_1 "task_pthread_1")
set(TASK_PTHREAD_2 "task_pthread_2")
set(TASK_PTHREAD_3 "task_pthread_3")

set(TASK_PTHREAD_1_SRC
    task_1.cpp
)

set(TASK_PTHREAD_2_SRC
    task_2.cpp
)

set(TASK_PTHREAD_3_SRC
    task_3.cpp
)

add_executable(${TASK_PTHREAD_1} ${TASK_PTHREAD_1_SRC})
add_executable(${TASK_PTHREAD_2} ${TASK_PTHREAD_2_SRC})
add_executable(${TASK_PTHREAD_3} ${TASK_PTHREAD_3_SRC})
