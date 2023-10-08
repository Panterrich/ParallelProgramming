project(task_initial_omp)

set(TASK_INITIAL_OMP_1 "task_initial_omp_1")
set(TASK_INITIAL_OMP_2 "task_initial_omp_2")
set(TASK_INITIAL_OMP_3 "task_initial_omp_3")

set(TASK_INITIAL_OMP_1_SRC
    task_1.cpp
)

set(TASK_INITIAL_OMP_2_SRC
    task_2.cpp
)

set(TASK_INITIAL_OMP_3_SRC
    task_3.cpp
)

add_executable(${TASK_INITIAL_OMP_1} ${TASK_INITIAL_OMP_1_SRC})
add_executable(${TASK_INITIAL_OMP_2} ${TASK_INITIAL_OMP_2_SRC})
add_executable(${TASK_INITIAL_OMP_3} ${TASK_INITIAL_OMP_3_SRC})
