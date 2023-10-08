#include <omp.h>
#include <stdio.h>

static constexpr unsigned int kNumThreads = 4;

unsigned int b = 0xBAD;
#pragma omp threadprivate(b)

int main()
{
//==============================================================================
/**
 * @brief This application declares a global variable and specifies it as
 * threadprivate. This variable is then passed a copyin to the first parallel
 * region. In that region, the master thread modifies its value but other
 * threads will not see the update until the second parallel region; where the
 * variable will be passed as copyin again.
 **/
	#pragma omp parallel copyin(b) num_threads(kNumThreads)
	{
		#pragma omp master
		{
			b = 0xFACE;
			printf("[First  parallel region] Master thread changes the value of b to %#X.\n", b);
		}

		#pragma omp barrier

		printf("[First  parallel region] Thread #%-2d: b = %#X.\n", omp_get_thread_num(), b);
	}

	#pragma omp parallel copyin(b) num_threads(kNumThreads)
	{
		printf("[Second parallel region] Thread #%-2d: b = %#X.\n", omp_get_thread_num(), b);
	}

    printf("\n\n\n");

//==============================================================================
/**
 * @brief This application passes a variable as firstprivate to a parallel
 * region. Then, a single construct receives this variable as a copyprivate and
 * modifies its values. All threads print the value of their own copy before and
 * after the single construct. Although each thread has its own copy, the
 * copyprivate will have broadcasted the new value to all threads after the
 * single construct.
 **/
	unsigned int a = 0xEDA;

	#pragma omp parallel firstprivate(a) num_threads(kNumThreads)
    {
		printf("[First  barrier section] Thread #%-2d: a = %#X.\n", omp_get_thread_num(), a);

		#pragma omp barrier

		#pragma omp single copyprivate(a)
		{
			a = 0xDEAD;
			printf("[       barrier section] Thread #%-2d executes the single construct and changes a to %#X.\n", omp_get_thread_num(), a);
		}

        printf("[Second barrier section] Thread #%-2d: a = %#X.\n", omp_get_thread_num(), a);
	}
//==============================================================================
}
