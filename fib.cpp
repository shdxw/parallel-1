//#include "fib.h"
//
//unsigned Fibonacci(unsigned n){
//    if (n <= 2)
//        return 1;
//    return Fibonacci(n-1) + Fibonacci(n-2);
//}
//
//unsigned Fibonacci_omp(unsigned n){
//    if (n <= 2)
//        return 1;
//    unsigned x1, x2;
//#pragma omp task
//    {
//        x1 = Fibonacci_omp(n-1);
//    };
//#pragma omp task
//    {
//        x2 = Fibonacci_omp(n-2);
//    };
//#pragma omp taskwait
//    return x1 + x2;
//}
//
//std::future<unsigned> async_Fibonacci(unsigned n)
//{
//    if (n <= 2) {
//        auto fut = std::async([=]() { return (unsigned)1; });
//        return fut;
//    }
//    auto fut = std::async([=]() {
//        std::future<unsigned> a = async_Fibonacci(n - 1);
//        std::future<unsigned> b = async_Fibonacci(n - 2);
//        unsigned c = a.get() + b.get();
//        return c;
//    });
//    return fut;
//}
//
//unsigned Fibonacci_sch_omp(unsigned n){
//    unsigned acc [n];
//#pragma omp for schedule(dynamic)
//        for (int i=0; i<n; i++){
//            if (i<=1) { acc[i] = 1; }
//            else{
//                acc[i] = acc[i-1] + acc[i-2];
//            }
//        }
//    unsigned res = 0;
//    for (int i=0; i<n; i++){
//        res += acc[i];
//    }
//return res;
//}