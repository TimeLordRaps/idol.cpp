// Timing helpers for the command-line tools. Not part of the library.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>

#include <ctime>
#endif

namespace idol::tools {

// CPU time of this process, all threads together, in milliseconds. Unlike wall time it does not count the time a
// thread spends waiting for a core, so it stays steady when the machine is busy.
inline double process_cpu_ms() {
#ifdef _WIN32
    FILETIME created, exited, kernel, user;
    GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user);
    auto to_ms = [](const FILETIME& f) {
        return static_cast<double>((static_cast<std::uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime) / 10000.0;
    };
    return to_ms(kernel) + to_ms(user);
#else
    timespec ts{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) / 1e6;
#endif
}

// The most memory this process has had resident at once, in megabytes (10^6 bytes).
inline double peak_memory_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters));
    return static_cast<double>(counters.PeakWorkingSetSize) / 1e6;
#else
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return static_cast<double>(usage.ru_maxrss) / 1e6;  // bytes
#else
    return static_cast<double>(usage.ru_maxrss) / 1e3;  // kilobytes
#endif
#endif
}

// Median of a non-empty list. Takes a copy because it sorts.
inline double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

}  // namespace idol::tools
