#define STEPS 100000000
#define MIN 1
#define MAX 300
#define SEED 100
#define CACHE_LINE 64u

#include <thread>
#include <mutex>
#include <vector>
#include <omp.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include "barrier.h"
#include <algorithm>
#include "iostream"
#include "fib.h"
#include "randomize.h"

using namespace std;

typedef double (*f_t)(double);
typedef double (*fib_t)(double);

typedef double (*I_t)(f_t, double, double);
typedef unsigned (*F_t)(unsigned);
typedef double (*R_t)(unsigned*, size_t);

typedef struct experiment_result {
    double result;
    double time_ms;
} experiment_result;

typedef struct partial_sum_t_ {
    alignas(64) double value;
} partial_sum_t_;
typedef struct partial_sum_t_for_rand {
    alignas(64) unsigned value;
} partial_sum_t_for_rand_;

#if defined(__GNUC__) && __GNUC__ <= 10
namespace std {
    constexpr size_t hardware_constructive_interference_size = 64u;
    constexpr size_t hardware_destructive_interference_size = 64u;
}
#endif

size_t ceil_div(size_t x, size_t y) {
    return (x + y - 1) / y;
}

double f(double x) {
    return x * x;
}

unsigned g_num_threads = thread::hardware_concurrency();

void set_num_threads(unsigned T) {
    omp_set_num_threads(T);
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

experiment_result run_experiment_random(R_t R) {
    size_t len = 100000;
    unsigned arr[len];
    unsigned seed = 100;

    double t0 = omp_get_wtime();
    double Res = R((unsigned *)&arr, len);
    double t1 = omp_get_wtime();
    return {Res, t1 - t0};
}

void show_experiment_result(I_t I) {
    double T1;
    printf("%10s\t%10s\t%10sms\t%13s\n", "Threads", "Result", "Time", "Acceleration");
    for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
        experiment_result R;
        set_num_threads(T);
        R = run_experiment(I);
        if (T == 1) {
            T1 = R.time_ms;
        }
        printf("%10u\t%10g\t%10g\t%10g\n", T, R.result, R.time_ms, T1/R.time_ms);
    };
}

void show_experiment_result_Rand(R_t Rand) {
    double T1;
    uint64_t a = 6364136223846793005;
    unsigned b = 1;

    double dif = 0;
    double avg = (MAX + MIN)/2;

    printf("%10s\t%10s\t%10s\t%10s\t%10s\n", "Threads", "Result", "Avg", "Difference", "Acceleration");
    for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
        set_num_threads(T);
        experiment_result R = run_experiment_random(Rand);
        if (T == 1) {
            T1 = R.time_ms;
        }
        dif = avg - R.result;
        printf("%10u\t%10g\t%10g\t%10g\t%10g\n", T, R.result, avg, dif, T1/R.time_ms);
    };
}

void show_experiment_result_json(I_t I, string name) {
    printf("{\n\"name\": \"%s\",\n", name.c_str());
    printf("\"points\": [\n");
    for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
        experiment_result R;
        set_num_threads(T);
        R = run_experiment(I);
        printf("{ \"x\": %8u, \"y\": %8g}", T, R.time_ms);
        if (T < omp_get_num_procs()) printf(",\n");
    }
    printf("]\n}");
}

double Integrate(f_t f, double a, double b) {//integrateArr

    unsigned T;
    double global_result = 0;
    double dx = (b - a) / STEPS;
    double *acc;

#pragma omp parallel shared(acc, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) get_num_threads();
            acc = (double *) calloc(T, sizeof(double));
        }
        for (unsigned i = t; i < STEPS; i += T) {
            acc[t] += f(dx * i + a);
        }
    }
    for (unsigned i = 0; i < T; ++i) {
        global_result += acc[i];
    }
    free(acc);

    return global_result * dx;
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
                scoped_lock lck{mtx};
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

double integrate_reduction(f_t f, double a, double b) {
    double Result = 0;
    double dx = (b - a) / STEPS;

#pragma omp parallel for reduction(+:Result)
    for (int i = 0; i < STEPS; i++)
        Result += f(dx * i + a);

    Result *= dx;
    return Result;
}

double integrate_ps_align_omp(f_t f, double a, double b) {
    double global_result = 0;
    partial_sum_t_ *partial_sum;
    double dx = (b - a) / STEPS;
    unsigned T;
#pragma omp parallel shared(partial_sum, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma opm single
        {
            T = (unsigned) omp_get_num_threads();
            partial_sum = (partial_sum_t_ *) aligned_alloc(CACHE_LINE, T * sizeof(partial_sum_t_));
            memset(partial_sum, 0, T * sizeof(*partial_sum));
        };
        for (unsigned i = t; i < STEPS; i += T)
            partial_sum[t].value += f(dx * i + a);
    }
    for (unsigned i = 0; i < T; ++i)
        global_result += partial_sum[i].value;

    free(partial_sum);
    return global_result * dx;
}

double integrate_ps_cpp(f_t f, double a, double b) {
    using namespace std;
    double dx = (b - a) / STEPS;
    double result = 0;
    unsigned T = get_num_threads();
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

template <class ElementType, class BinaryFn>
ElementType reduce_vector(const ElementType* V, size_t n, BinaryFn f, ElementType zero)
{
    unsigned T = get_num_threads();
    struct reduction_partial_result_t
    {
        alignas(hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =
            vector<reduction_partial_result_t>(thread::hardware_concurrency(),
                                                            reduction_partial_result_t{zero});
    constexpr size_t k = 2;
    barrier bar {T};

    auto thread_proc = [=, &bar](unsigned t)
    {
        auto K = ceil_div(n, k);
        size_t Mt = K / T;
        size_t it1 = K % T;

        if(t < it1)
        {
            it1 = ++Mt * t;
        }
        else
        {
            it1 = Mt * t + it1;
        }
        it1 *= k;
        size_t mt = Mt * k;
        size_t it2 = it1 + mt;

        ElementType accum = zero;
        for(size_t i = it1; i < it2; i++)
            accum = f(accum, V[i]);

        reduction_partial_results[t].value = accum;

        for(std::size_t s = 1, s_next = 2; s < T; s = s_next, s_next += s_next)
        {
            bar.arrive_and_wait();
            if(((t % s_next) == 0) && (t + s < T))
                reduction_partial_results[t].value = f(reduction_partial_results[t].value,
                                                       reduction_partial_results[t + s].value);
        }
    };

    vector<thread> threads;
    for(unsigned t = 1; t < T; t++)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for(auto& thread : threads)
        thread.join();

    return reduction_partial_results[0].value;
}

template <class ElementType, class UnaryFn, class BinaryFn>
// requires {
//     is_invocable_r_v<UnaryFn, ElementType, ElementType> &&
//     is_invocable_r_v<BinaryFn, ElementType, ElementType, ElementType>
// }
ElementType reduce_range(ElementType a, ElementType b, size_t n, UnaryFn get, BinaryFn reduce_2, ElementType zero)
{
    unsigned T = get_num_threads();
    struct reduction_partial_result_t
    {
        alignas(hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =
            vector<reduction_partial_result_t>(thread::hardware_concurrency(), reduction_partial_result_t{zero});

    barrier bar{T};
    constexpr size_t k = 2;
    auto thread_proc = [=, &bar](unsigned t)
    {
        auto K = ceil_div(n, k);
        double dx = (b - a) / n;
        size_t Mt = K / T;
        size_t it1 = K % T;

        if(t < it1)
        {
            it1 = ++Mt * t;
        }
        else
        {
            it1 = Mt * t + it1;
        }
        it1 *= k;
        size_t mt = Mt * k;
        size_t it2 = it1 + mt;

        ElementType accum = zero;
        for(size_t i = it1; i < it2; i++)
            accum = reduce_2(accum, get(a + i*dx));

        reduction_partial_results[t].value = accum;

        for(size_t s = 1, s_next = 2; s < T; s = s_next, s_next += s_next)
        {
            bar.arrive_and_wait();
            if(((t % s_next) == 0) && (t + s < T))
                reduction_partial_results[t].value = reduce_2(reduction_partial_results[t].value,
                                                              reduction_partial_results[t + s].value);
        }
    };

    vector<thread> threads;
    for (unsigned t = 1; t < T; t++)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for (auto &thread: threads)
        thread.join();
    return reduction_partial_results[0].value;
}

double integrate_reduction(double a, double b, f_t f) {
    return reduce_range(a, b, STEPS, f, [](auto x, auto y) { return x + y; }, 0.0) * ((b - a) / STEPS);
}

//---Randomize----------------------------------------------------------------------------------------------------------

double randomize_arr_single(unsigned* V, size_t n){
    uint64_t a = 6364136223846793005;
    unsigned b = 1;
    uint64_t prev = SEED;
    uint64_t sum = 0;

    for (unsigned i=0; i<n; i++){
        uint64_t cur = a*prev + b;
        V[i] = (cur % (MAX - MIN + 1)) + MIN;
        prev = cur;
        sum +=V[i];
    }

    return (double)sum/(double)n;
}

uint64_t* getLUTA(unsigned size, uint64_t a){
    uint64_t res[size+1];
    res[0] = 1;
    for (unsigned i=1; i<=size; i++) res[i] = res[i-1] * a;
    return res;
}

uint64_t* getLUTB(unsigned size, uint64_t* a, uint64_t b){
    uint64_t res[size];
    res[0] = b;
    for (unsigned i=1; i<size; i++){
        uint64_t acc = 0;
        for (unsigned j=0; j<=i; j++){
            acc += a[j];
        }
        res[i] = acc*b;
    }
    return res;
}

uint64_t getA(unsigned size, uint64_t a){
    uint64_t res = 1;
    for (unsigned i=1; i<=size; i++) res = res * a;
    return res;
}

uint64_t getB(unsigned size, uint64_t a){
    uint64_t* acc = (uint64_t *) calloc(size+1, sizeof(uint64_t));
//    uint64_t* acc = new uint64_t(size);
    uint64_t res = 1;
    acc[0] = 1;
    for (unsigned i=1; i<=size; i++){
        for (unsigned j=0; j<i; j++){
            acc[i] = acc[j] * a;
        }
        res += acc[i];
    }
    free(acc);
    return res;
}

double randomize_arr_fs(unsigned* V, size_t n){
    uint64_t a = 6364136223846793005;
    unsigned b = 1;
    unsigned T;
//    uint64_t* LUTA;
//    uint64_t* LUTB;
    uint64_t LUTA;
    uint64_t LUTB;
    uint64_t sum = 0;

#pragma omp parallel shared(V, T, LUTA, LUTB)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) get_num_threads();
//            LUTA = getLUTA(n, a);
//            LUTB = getLUTB(n, LUTA, b);
            LUTA = getA(T, a);
            LUTB = getB((T - 1), a)*b;
        }
        uint64_t prev = SEED;
        uint64_t cur;

        for (unsigned i=t; i<n; i += T){
            if (i == t){
                cur = getA(i+1, a)*prev + getB(i, a) * b;
            } else {
                cur = LUTA*prev + LUTB;
            }
//            cur = LUTA[i+1]*prev + LUTB[i];
            V[i] = (cur % (MAX - MIN + 1)) + MIN;
            prev = cur;
        }
    }

    for (unsigned i=0; i<n;i++)
        sum += V[i];

    return (double)sum/(double)n;
}

//---Fibonacci----------------------------------------------------------------------------------------------------------

unsigned Fibonacci(unsigned n){
    if (n <= 2)
        return 1;
    return Fibonacci(n-1) + Fibonacci(n-2);
}

unsigned Fibonacci_omp(unsigned n){
    if (n <= 2)
        return 1;
    unsigned x1, x2;
#pragma omp task shared(x1)
    {
        x1 = Fibonacci_omp(n-1);
    };
#pragma omp task shared(x2)
    {
        x2 = Fibonacci_omp(n-2);
    };
#pragma omp taskwait
    return x1 + x2;
}

#include <future>
std::future<unsigned> async_Fibonacci(unsigned n)
{
    if (n <= 2) {
        auto fut = std::async([=]() { return (unsigned)1; });
        return fut;
    }
    auto fut = std::async([=]() {
        std::future<unsigned> a = async_Fibonacci(n - 1);
        std::future<unsigned> b = async_Fibonacci(n - 2);
        unsigned c = a.get() + b.get();
        return c;
    });
    return fut;
}

unsigned Fibonacci_sch_omp(unsigned n){
    unsigned acc [n];
#pragma omp for schedule(dynamic)
    for (int i=0; i<n; i++){
        if (i<=1) { acc[i] = 1; }
        else{
            acc[i] = acc[i-1] + acc[i-2];
        }
    }
    unsigned res = 0;
    for (int i=0; i<n; i++){
        res += acc[i];
    }
    return res;
}

experiment_result run_experiment_fib(F_t f) {
    double t0 = omp_get_wtime();
    double R = f(10);
    double t1 = omp_get_wtime();
    return {R, t1 - t0};
}

void show_experiment_result_Fib(F_t f) {
    double T1;
    printf("%10s\t%10s\t%10sms\t%13s\n", "Threads", "Result", "Time", "Acceleration");
    for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
        experiment_result R;
        set_num_threads(T);
        R = run_experiment_fib(f);
        if (T == 1) {
            T1 = R.time_ms;
        }
        printf("%10u\t%10g\t%10g\t%10g\n", T, R.result, R.time_ms, T1/R.time_ms);
    };
}

int main() {
    experiment_result p;

//    printf("Integrate with single(omp)\n");
//    show_experiment_result(Integrate);
//    printf("Integrate with critical sections(omp)\n");
//    show_experiment_result(integrate_crit);
//    printf("Integrate with mutex(cpp)\n");
//    show_experiment_result(integrate_cpp_mtx);
//    printf("Integrate reduction (omp)\n");
//    show_experiment_result(integrate_reduction);
//    printf("Integrate aligned array with partial sums(omp)\n");
//    show_experiment_result(integrate_ps_align_omp);
//    printf("Integrate with partial sums(cpp)\n");
//    show_experiment_result(integrate_ps_cpp);
//    printf("Integrate with atomic operations(cpp)\n");
//    show_experiment_result(integrate_cpp_atomic);

//    printf("{\n\"series\": [\n");
//    show_experiment_result_json(Integrate, "Integrate");
//    printf(",");
//    show_experiment_result_json(integrate_crit, "integrate_crit");
//    printf(",");
//    show_experiment_result_json(integrate_cpp_mtx, "integrate_cpp_mtx");
//    printf(",");
//    show_experiment_result_json(integrate_ps, "integrate_ps_cpp");
//    printf(",");
//    show_experiment_result_json(integratePS, "integratePS");
//    printf("]}");
//    show_experiment_result_json(integrate_cpp_atomic, "integrate_cpp_atomic");
//    printf("]}");

//    unsigned V[16];
//    double a = -1, b = 1;
//    double dx = (b-a)/ STEPS;
//    for (unsigned i = 1; i <= size(V); i++) {
//        V[i] = i + 1;
//        cout << "Average: " << reduce_vector(V, size(V), [](auto x, auto y) {  return x + y; }, 0u) / size(V) << '\n';
//    }
//    for (unsigned i = 1; i <= size(V); i++) {
//        cout << "Average: " << reduce_range(-1, 1, STEPS, f, [](auto x, auto y) {return x + y;}, 0) << '\n';
//    }
//
//    unsigned param = 10;
//    unsigned fibonacci = Fibonacci(param);
//    cout << fibonacci << endl;
//    unsigned fibonacci_omp = Fibonacci_omp(param);
//    cout << fibonacci_omp << endl;
//    unsigned fibonacci_sch_omp = Fibonacci_sch_omp(param);
//    cout << fibonacci_sch_omp << endl;
//    unsigned asynciBbonacci = async_Fibonacci(param).get();
//    cout << asynciBbonacci << endl;
//
//    printf("Fib omp\n");
//    show_experiment_result_Fib(Fibonacci_omp);
//    printf("Fib schedule omp\n");
//    show_experiment_result_Fib(Fibonacci_sch_omp);
//    printf("Fib\n");
//    show_experiment_result_Fib(Fibonacci);


    printf("Rand single\n");
    show_experiment_result_Rand(randomize_arr_single);
    printf("Rand omp fs\n");
    show_experiment_result_Rand(randomize_arr_fs);

    size_t len = 2000000;
    unsigned arr[len];

    cout << randomize_arr_single(arr, len) << endl;
    cout << randomize_arr_fs(arr, len) << endl;

    return 0;
}
