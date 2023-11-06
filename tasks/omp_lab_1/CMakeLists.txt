project(omp_lab_1)

find_package(benchmark REQUIRED)

set(LAB_OMP "lab_omp")
set(LAB_MPI_0 "lab_mpi_0")
set(LAB_MPI_1 "lab_mpi_1")
set(LAB_MPI_3 "lab_mpi_3")

set(LAB_OMP_SRC
    lab_omp.cpp
)

set(LAB_MPI_0_SRC
    mpi_0.cpp
)

set(LAB_MPI_1_SRC
    mpi_1.cpp
)

set(LAB_MPI_3_SRC
    mpi_3.cpp
)

add_executable(${LAB_OMP} ${LAB_OMP_SRC})
add_executable(${LAB_MPI_0} ${LAB_MPI_0_SRC})
add_executable(${LAB_MPI_1} ${LAB_MPI_1_SRC})
add_executable(${LAB_MPI_3} ${LAB_MPI_3_SRC})

target_link_libraries(${LAB_OMP} benchmark::benchmark)


