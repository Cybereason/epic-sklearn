#pragma once

#include <mutex>
#include <atomic>
#include <thread>
#include <vector>
#include <exception>
#include <functional>


/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc.)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The function is borrowed from nmslib.
 */
inline void parallel_for(size_t start, size_t end, const std::function<void(size_t, size_t)> &fn, size_t n_threads=0) {
    if (n_threads == 0)
        n_threads = std::thread::hardware_concurrency();

    if (n_threads == 1 || end - start < 2) {
        for (size_t i = start; i < end; ++i)
            fn(i, 0);
        return;
    }

    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);

    // keep track of exceptions in threads
    // https://stackoverflow.com/a/32428427/1713196
    std::exception_ptr last_exception = nullptr;
    std::mutex last_except_mutex;

    for (size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
        threads.emplace_back([&, thread_id] {
            while (true) {
                size_t i = current.fetch_add(1);
                if (i >= end)
                    break;
                try {
                    fn(i, thread_id);
                } catch (...) {
                    std::unique_lock<std::mutex> last_except_lock(last_except_mutex);
                    last_exception = std::current_exception();
                    /*
                     * This will work even when current is the largest value that
                     * size_t can fit, because fetch_add returns the previous value
                     * before the increment (what will result in overflow
                     * and produce 0 instead of current + 1).
                     */
                    current = end;
                    break;
                }
            }
        });
    }
    for (auto &thread : threads)
        thread.join();
    if (last_exception)
        std::rethrow_exception(last_exception);
}
