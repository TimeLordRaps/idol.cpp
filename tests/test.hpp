// A small dependency-free test harness. TEST(suite, name) registers a function. The CHECK macros record a failure and
// let the test carry on, so one run reports every problem it finds.
#pragma once

#include <cmath>
#include <exception>
#include <string>

namespace idol::test {

using TestFn = void (*)();

struct Registrar {
    Registrar(const char* suite, const char* name, TestFn body);
};

// Records one failed check. The macros below call this.
void report_failure(const char* file, int line, const std::string& message);

}  // namespace idol::test

#define TEST(suite, name)                                                                        \
    static void test_##suite##_##name();                                                         \
    static const ::idol::test::Registrar registrar_##suite##_##name(#suite, #name,               \
                                                                    &test_##suite##_##name);     \
    static void test_##suite##_##name()

#define CHECK(condition)                                                                         \
    do {                                                                                         \
        if (!(condition)) ::idol::test::report_failure(__FILE__, __LINE__, "CHECK(" #condition ")"); \
    } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                                  \
    do {                                                                                         \
        const double idol_test_actual = static_cast<double>(actual);                             \
        const double idol_test_expected = static_cast<double>(expected);                         \
        if (!(std::fabs(idol_test_actual - idol_test_expected) <= (tolerance)))                  \
            ::idol::test::report_failure(__FILE__, __LINE__,                                     \
                "CHECK_NEAR(" #actual ", " #expected ") got " + std::to_string(idol_test_actual) \
                    + ", expected " + std::to_string(idol_test_expected));                       \
    } while (0)

// Passes when the statement throws an exception derived from std::exception.
#define CHECK_THROWS(statement)                                                                  \
    do {                                                                                         \
        bool idol_test_threw = false;                                                            \
        try {                                                                                    \
            statement;                                                                           \
        } catch (const std::exception&) {                                                        \
            idol_test_threw = true;                                                              \
        }                                                                                        \
        if (!idol_test_threw)                                                                    \
            ::idol::test::report_failure(__FILE__, __LINE__, "expected an exception from " #statement); \
    } while (0)
