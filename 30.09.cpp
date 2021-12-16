#define STEPS 100000000
#define CACHE_LINE 64u

#include <thread>
#include <mutex>
#include <vector>
#include <omp.h>
#include <atomic>

#include <cstdlib>
#include <thread>
#include <vector>
//#include <barrier>

#if defined(__GNUC__) && __GNUC__ <= 10
namespace std {
    constexpr std::size_t hardware_constructive_interference_size = 64u;
    constexpr std::size_t hardware_destructive_interference_size = 64u;
}
#endif

std::size_t ceil_div(std::size_t x, std::size_t y) {
    return (x + y - 1) / y;
}

double f(double x) {
    return x * x;
}

unsigned g_num_threads = std::thread::hardware_concurrency();

typedef double (*f_t)(double);

typedef double (*I_t)(f_t, double, double);

typedef struct experiment_result {
    double result;
    double time_ms;
} experiment_result;

typedef struct partial_sum_t_ {
    alignas(64) double value;
} partial_sum_t_;

void set_num_threads(unsigned T) {
    g_num_threads = T;
}

unsigned get_num_threads() {
    return g_num_threads;
}

experiment_result run_experiment(I_t I) {
    double t0 = omp_get_wtime();
    double R = I(f, -1, 1);
    double t1 = omp_get_wtime();
    return {R, t1 - t0};
}

void show_experiment_result(I_t I) {
    double T1;
    printf("%10s\t%10s\t%10sms\t%13s\n", "Threads", "Result", "Time", "Acceleration");
    for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
        experiment_result R;
        omp_set_num_threads(T);
        R = run_experiment(I);
        if (T == 1) {
            T1 = R.time_ms;
        }
        printf("%10u\t%10g\t%10g\t%10g\n", T, R.result, R.time_ms, T1/R.time_ms);
    };
}

void show_experiment_result_json(I_t I, std::string name) {
    printf("{\n\"name\": \"%s\",\n", name.c_str());
    printf("\"points\": [\n");
    for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
        experiment_result R;
        omp_set_num_threads(T);
        R = run_experiment(I);
        printf("{ \"x\": %8u, \"y\": %8g}", T, R.time_ms);
        if (T < omp_get_num_procs()) printf(",\n");
    }
    printf("]\n}");
}

double Integrate(f_t f, double a, double b) {//IntegrateFalseSharingOMP
    unsigned T;
    double Result = 0;
    double dx = (b - a) / STEPS;
    double *Accum;
#pragma omp parallel shared(Accum, T)
    {
        unsigned int t = (unsigned int) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            Accum = (double *) calloc(T, sizeof(double));
        }
        for (unsigned i = t; i < STEPS; i += T)
            Accum[t] += f(dx * i + a);
    }
    for (unsigned i = 0; i < T; ++i)
        Result += Accum[i];
    free(Accum);
    return Result * dx;
}

double integrate_cpp_mtx(f_t f, double a, double b) {
    using namespace std;
    unsigned T = get_num_threads();
    vector<thread> threads;
    mutex mtx;
    double Result = 0;
    double dx = (b - a) / STEPS;
    for (unsigned t = 0; t < T; t++) {
        threads.emplace_back([=, &Result, &mtx]() {
            double R = 0;
            for (unsigned i = t; i < STEPS; i += T)
                R += f(dx * i + a);
            {
                std::scoped_lock lck{mtx};
                Result += R;
            }
        });
    }
    for (auto &thr: threads) thr.join();
    return Result * dx;
}

double integrate_crit(f_t f, double a, double b) { //IntegrateParallelOMP
    double Result = 0;
    double dx = (b - a) / STEPS;
#pragma omp parallel shared(Result)
    {
        double R = 0;
        unsigned t = omp_get_thread_num();
        unsigned T = (unsigned) omp_get_num_threads();
        for (unsigned i = t; i < STEPS; i += T)
            R += f(i * dx + a);
#pragma omp critical
        Result += R;
    }
    return Result * dx;
}

double IntegrateReductionOMP(f_t f, double a, double b)
{
    double Result = 0;
    double dx = (b-a)/STEPS;

#pragma omp parallel for reduction(+:Result)
    for(int i = 0; i < STEPS; i++)
        Result += f(dx*i + a);

    Result *= dx;
    return Result;
}

double integrate_ps(f_t f, double a, double b) {//IntegrateAlignOMP
    double global_result = 0;
    partial_sum_t_ *partial_sum;
    double dx = (b - a) / STEPS;
    unsigned T;
#pragma omp parallel shared(partial_sum, T)
    {
#pragma opm single
        {
            T = (unsigned) omp_get_num_threads();
            partial_sum = (partial_sum_t_ *) aligned_alloc(CACHE_LINE, T * sizeof(partial_sum_t_));
        };
        unsigned t = (unsigned) omp_get_thread_num();
        double R = 0;
        for (unsigned i = t; i < STEPS; i += T)
            partial_sum[t].value += f(dx * i + a);
    }
    for (unsigned i = 0; i < T; ++i)
        global_result += partial_sum[i].value;
    free(partial_sum);
    return global_result * dx;
}

double integratePS(f_t f, double a, double b) {
    using namespace std;
    double dx = (b - a) / STEPS;
    double result = 0;
    unsigned T = thread::hardware_concurrency();
    auto vec = vector(T, partial_sum_t_{0.0});
    vector<thread> thread_vec;
    auto thread_proc = [=, &vec](auto t) {
        for (unsigned i = t; i < STEPS; i += T)
            vec[t].value += f(dx * i + a);
    };
    for (unsigned t = 1; t < T; t++) {
        thread_vec.emplace_back(thread_proc, t);
    }
    thread_proc(0);
    for (auto &thread: thread_vec) {
        thread.join();
    }
    for (auto elem: vec) {
        result += elem.value;
    }
    return result * dx;
}

double integrate_cpp_atomic(f_t f, double a, double b) {
    using namespace std;
    vector<thread> threads;
    int T = get_num_threads();
    atomic<double> Result{0.0};
    double dx = (b - a) / STEPS;
    auto fun = [dx, &Result, a, b, f, T](auto t) {
        double R = 0;
        for (unsigned i = t; i < STEPS; i += T)
            R += f(dx * i + a);
        Result = Result + R;
    };
    for (unsigned int t = 1; t < T; t++) {
        threads.emplace_back(fun, t);
    }
    fun(0);
    for (auto &thr: threads) thr.join();
    return Result * dx;
}

//template<class ElementType, class BinaryFn>
//ElementType reduce_vector(const ElementType *V, std::size_t n, BinaryFn f, ElementType zero) {
//    unsigned T = get_num_threads();
//    struct reduction_partial_result_t {
//        alignas(std::hardware_destructive_interference_size) ElementType value;
//    };
//    static auto reduction_partial_results = std::vector<reduction_partial_result_t>(T,reduction_partial_result_t{zero});
//
//    constexpr std::size_t k = 2;
//    auto thread_proc = [=](unsigned t) {
//        auto K = ceil_div(n, k);
//        std::size_t Mt = K / T;
//        std::size_t it1 = K % T;
//
//        if (t < (K % T)) {
//            it1 = ++Mt * t;
//        } else {
//            it1 = Mt * it1 + t;
//        }
//        it1 *= k;
//        std::size_t mt = Mt * k;
//        std::size_t it2 = it1 + mt;
//
//        ElementType accum = zero;
//        for (std::size_t i = it1; i < it2; i++)
//            accum = f(accum, V[i]);
//
//        reduction_partial_results[t].value = accum;
//    };
//
//    auto thread_proc_2_ = [=](unsigned t, std::size_t s) {
//        if (((t % (s * k)) == 0) && (t + s < T))
//            reduction_partial_results[t].value = f(reduction_partial_results[t].value,
//                                                   reduction_partial_results[t + s].value);
//    };
//
//    std::vector<std::thread> threads;
//    for (unsigned t = 1; t < T; t++)
//        threads.emplace_back(thread_proc, t);
//    thread_proc(0);
//
//    for (auto &thread: threads)
//        thread.join();
//
//    std::size_t s = 1;
//    while (s < T) {
//        for (unsigned t = 1; t < T; t++) {
//            threads[t - 1] = std::thread(thread_proc_2_, t, s);
//        }
//        thread_proc_2_(0, s);
//        s *= k;
//
//        for (auto &thread: threads)
//            thread.join();
//    }
////    printf("{%8u} - ", reduction_partial_results[0].value);
//    return reduction_partial_results[0].value;
//}

#include <type_traits>

#include <omp.h>
#include <cstdlib>
#include <thread>
#include <vector>
#include <barrier>

template <class ElementType, class BinaryFn>
ElementType reduce_vector(const ElementType* V, std::size_t n, BinaryFn f, ElementType zero)
{
    unsigned T = get_num_threads();
    struct reduction_partial_result_t
    {
        alignas(std::hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =
            std::vector<reduction_partial_result_t>(std::thread::hardware_concurrency(),
                                                            reduction_partial_result_t{zero});
    constexpr std::size_t k = 2;
    std::barrier<> bar {T};

    auto thread_proc = [=, &bar](unsigned t)
    {
        auto K = ceil_div(n, k);
        std::size_t Mt = K / T;
        std::size_t it1 = K % T;

        if(t < it1)
        {
            it1 = ++Mt * t;
        }
        else
        {
            it1 = Mt * t + it1;
        }
        it1 *= k;
        std::size_t mt = Mt * k;
        std::size_t it2 = it1 + mt;

        ElementType accum = zero;
        for(std::size_t i = it1; i < it2; i++)
            accum = f(accum, V[i]);

        reduction_partial_results[t].value = accum;

        std::size_t s = 1;
        while(s < T)
        {
            bar.arrive_and_wait();
            if((t % (s * k)) && (t + s < T))
            {
                reduction_partial_results[t].value = f(reduction_partial_results[t].value,
                                                       reduction_partial_results[t + s].value);
                s *= k;
            }
        }
    };

    std::vector<std::thread> threads;
    for(unsigned t = 1; t < T; t++)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for(auto& thread : threads)
        thread.join();

    return reduction_partial_results[0].value;
}

template <class ElementType, class UnaryFn, class BinaryFn>
#if 0
requires {
    std::is_invocable_r_v<UnaryFn, ElementType, ElementType> &&
    std::is_invocable_r_v<BinaryFn, ElementType, ElementType, ElementType>
}
#endif
ElementType reduce_range(ElementType a, ElementType b, std::size_t n, UnaryFn get, BinaryFn reduce_2, ElementType zero)
{
    unsigned T = get_num_threads();
    struct reduction_partial_result_t
    {
        alignas(std::hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =
            std::vector<reduction_partial_result_t>(std::thread::hardware_concurrency(), reduction_partial_result_t{zero});

    std::barrier<> bar{T};
    constexpr std::size_t k = 2;
    auto thread_proc = [=, &bar](unsigned t)
    {
        auto K = ceil_div(n, k);
        double dx = (b - a) / n;
        std::size_t Mt = K / T;
        std::size_t it1 = K % T;

        if(t < it1)
        {
            it1 = ++Mt * t;
        }
        else
        {
            it1 = Mt * t + it1;
        }
        it1 *= k;
        std::size_t mt = Mt * k;
        std::size_t it2 = it1 + mt;

        ElementType accum = zero;
        for(std::size_t i = it1; i < it2; i++)
            accum = reduce_2(accum, get(a + i*dx));

        reduction_partial_results[t].value = accum;

        for(std::size_t s = 1, s_next = 2; s < T; s = s_next, s_next += s_next)
        {
            bar.arrive_and_wait();
            if(((t % s_next) == 0) && (t + s < T))
                reduction_partial_results[t].value = reduce_2(reduction_partial_results[t].value,
                                                              reduction_partial_results[t + s].value);
        }
    };

    std::vector<std::thread> threads;
    for(unsigned t = 1; t < T; t++)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for(auto& thread : threads)
        thread.join();
    return reduction_partial_results[0].value;
}

double integrate_reduction(double a, double b, f_t f){
    return reduce_range(a, b, STEPS, f, [](auto x, auto y) {return x + y;}, 0.0) * ((b - a) / STEPS);
}

#include <algorithm>
#include "iostream"

int main() {
    experiment_result p;

    printf("Integrate with single(omp)\n");
    show_experiment_result(Integrate);
    printf("Integrate with critical sections(omp)\n");
    show_experiment_result(integrate_crit);
    printf("Integrate with mutex(cpp)\n");
    show_experiment_result(integrate_cpp_mtx);
    printf("Integrate reduction (omp)\n");
    show_experiment_result(IntegrateReductionOMP);
    printf("Integrate with partial sums(omp)\n");
    show_experiment_result(integrate_ps);
    printf("Integrate with partial sums(cpp)\n");
    show_experiment_result(integratePS);
    printf("Integrate with atomic operations(cpp)\n");
    show_experiment_result(integrate_cpp_atomic);

//    printf("{\n\"series\": [\n");
//    show_experiment_result_json(Integrate, "Integrate");
//    printf(",");
//    show_experiment_result_json(integrate_crit, "integrate_crit");
//    printf(",");
//    show_experiment_result_json(integrate_cpp_mtx, "integrate_cpp_mtx");
//    printf(",");
//    show_experiment_result_json(integrate_ps, "integrate_ps");
//    printf(",");
//    show_experiment_result_json(integratePS, "integratePS");
//    printf("]}");
//    show_experiment_result_json(integrate_cpp_atomic, "integrate_cpp_atomic");
//    printf("]}");

    unsigned V[9];
    printf("%9s\t%9s\n", "V-size", "Average");
    for (unsigned i = 1; i <= std::size(V); i++) {
        V[i] = i + 1;
//        std::cout << i << "           " << reduce_vector(V, std::size(V), [](auto x, auto y) {  return x + y; }, 0u) / std::size(V) << '\n';
        std::cout << "Average: " << reduce_vector(V, std::size(V), [](auto x, auto y) {  return x + y; }, 0u) / std::size(V) << '\n';
         std::cout << "Average: " << reduce_range(1, 16, 10000, f, [](auto x, auto y) {return x + y;}, 0) << '\n';
    }

    return 0;
}
