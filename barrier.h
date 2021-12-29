#ifndef PARALLEL_BARRIER_H
#define PARALLEL_BARRIER_H

#include <condition_variable>
using namespace std;

class barrier {
    bool lock_oddity = false;
    unsigned T;
    const unsigned Tmax;
    condition_variable cv;
    mutex mtx;
public:
    barrier(unsigned T);
    void arrive_and_wait();
};

#endif //PARALLEL_BARRIER_H
