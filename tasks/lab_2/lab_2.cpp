#include "lab_2.h"

struct
{
    const ssize_t max_local_size  = 3;
    const ssize_t max_global_size = 50;

    double eps;

    std::stack<Task> stack;
    ssize_t n_task;

    double sum;

    size_t n_active;

    pthread_mutex_t mutex_sum;
    pthread_mutex_t mutex_stack;
    pthread_mutex_t mutex_active;

} shared;

int main(int argc, char* argv[])
{
    if (argc != 3) 
    {
        printf("Enter K number of threads and epsilon\n"
               "For example: ./a.out 10\n");
        return 0;
    }

    char* end = nullptr;
    unsigned long K = strtoul(argv[1], &end, 10);

    if ((errno == ERANGE) || (*end != '\0'))
        return 1;

    InitSharedMemory(std::atof(argv[2]));

    std::vector<pthread_t> pthreads(K);
    std::vector<ssize_t> tasks(K);

    auto start_time = std::chrono::high_resolution_clock::now();

    int error = 0;
    for (unsigned long i = 0; i < K; i++)
    {
        error = pthread_create(&pthreads[i], nullptr, routine_integrate, &tasks[i]);
        if (error)
        {
            perror("pthread_create");
            return 1;
        }
    }

    for (unsigned long i = 0; i < K; i++)
    {
        error = pthread_join(pthreads[i], nullptr);
        if (error)
        {
            perror("pthread_join");
            return 1;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    for (size_t i = 0; i < K; i++)
    {
        printf("tasks: %lu\n", tasks[i]);
    }

    printf("%.12lf\n", shared.sum);
    // std::cout << "Result: " << shared.sum << std::endl;
    //std::cout << std::fixed << "Result: " << std::setprecision(-std::ceil(std::log10(shared.eps))) << shared.sum << std::endl;
    std::cout << "Time: " << static_cast<double>(elapsed_ms.count()) / 1000.f << std::endl;

    DestroySharedMemory();

    return 0;
}

void* routine_integrate(void* arg)
{
    std::stack<Task> lstack;
    ssize_t n_task = 0;

    ssize_t tasks = 0;

    double sum = 0;

    while (1)
    {
        if (n_task == 0)
        {
            pthread_mutex_lock(&shared.mutex_stack);
            if (shared.n_task > 0)
            {
                shared.n_task--;
                lstack.push(shared.stack.top());
                n_task++;
                shared.stack.pop();
                
                pthread_mutex_lock(&shared.mutex_active);
                shared.n_active++;
                pthread_mutex_unlock(&shared.mutex_active);
            }
            pthread_mutex_unlock(&shared.mutex_stack);
        }

        if (n_task > 0)
        {
            struct Task task = lstack.top();
            lstack.pop();
            n_task--;

            double a  = task.a;
            double b  = task.b;
            double fa = task.fa;
            double fb = task.fb;
            double s  = task.s;

            while (1)
            {
                double c  = (a + b) / 2;
                double fc = Equation::Func::f(c);

                double s_ac = (fa + fc) * (c - a) / 2;
                double s_cb = (fc + fb) * (b - c) / 2;

                double s_acb = s_ac + s_cb;

                if (std::abs(s - s_acb) >= shared.eps * std::abs(s_acb) && std::abs(s - s_acb) > std::numeric_limits<double>::epsilon())
                {
                    lstack.push({a, c, fa, fc, s_ac});
                    n_task++;

                    a = c;
                    fa = fc;
                    s = s_cb;

                    if (n_task > shared.max_local_size && shared.n_task == 0)
                    {
                        while (n_task > 1 && shared.n_task < shared.max_global_size)
                        {
                            pthread_mutex_lock(&shared.mutex_stack);
                            shared.stack.push(lstack.top());
                            shared.n_task++;
                            pthread_mutex_unlock(&shared.mutex_stack);

                            lstack.pop();
                            n_task--;
                        }
                    }
                }
                else
                {
                    sum += s_acb;
                    tasks++;

                    if (n_task == 0)
                    {
                        pthread_mutex_lock(&shared.mutex_active);
                        shared.n_active--;
                        pthread_mutex_unlock(&shared.mutex_active);

                        break;
                    }

                    task = lstack.top();
                    lstack.pop();
                    n_task--;

                    a  = task.a;
                    b  = task.b;
                    fa = task.fa;
                    fb = task.fb;
                    s  = task.s;
                }
            }
        }

        if (n_task == 0)
        {
            pthread_mutex_lock(&shared.mutex_active);
            if (shared.n_active == 0 && shared.n_task == 0)
            {   
                pthread_mutex_unlock(&shared.mutex_active);
                break;
            }
            pthread_mutex_unlock(&shared.mutex_active);
        }
    }

    pthread_mutex_lock(&shared.mutex_sum);
    shared.sum += sum;
    pthread_mutex_unlock(&shared.mutex_sum);
    
    *(ssize_t*)(arg) = tasks;

    pthread_exit(NULL);
}

void InitSharedMemory(double eps)
{
    shared.eps = eps;
    
    pthread_mutex_init(&shared.mutex_stack,  nullptr);
    pthread_mutex_init(&shared.mutex_sum,    nullptr);
    pthread_mutex_init(&shared.mutex_active, nullptr);

    shared.sum = 0;

    shared.n_active = 0;

    struct Task init{};
    shared.stack.push(init);
    shared.n_task = 1;
}

void DestroySharedMemory()
{
    pthread_mutex_destroy(&shared.mutex_stack);
    pthread_mutex_destroy(&shared.mutex_sum);
    pthread_mutex_destroy(&shared.mutex_active);
}