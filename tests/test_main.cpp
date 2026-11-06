// Runs the registered tests. Given a suite name, only that suite runs. The exit status is 1 when a test fails or the
// suite name matches nothing.
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "test.hpp"

namespace idol::test {

namespace {

struct Entry {
    std::string suite;
    std::string name;
    TestFn body;
};

std::vector<Entry>& registry() {
    static std::vector<Entry> entries;
    return entries;
}

int failed_checks = 0;

}  // namespace

Registrar::Registrar(const char* suite, const char* name, TestFn body) {
    registry().push_back(Entry{suite, name, body});
}

void report_failure(const char* file, int line, const std::string& message) {
    ++failed_checks;
    std::printf("    %s:%d: %s\n", file, line, message.c_str());
}

}  // namespace idol::test

int main(int argc, char** argv) {
    const std::string suite = argc > 1 ? argv[1] : "";
    int ran = 0;
    int failed = 0;
    for (const idol::test::Entry& entry : idol::test::registry()) {
        if (!suite.empty() && entry.suite != suite) continue;
        ++ran;
        const int before = idol::test::failed_checks;
        try {
            entry.body();
        } catch (const std::exception& error) {
            idol::test::report_failure(__FILE__, __LINE__, std::string("unexpected exception: ") + error.what());
        }
        const bool ok = idol::test::failed_checks == before;
        if (!ok) ++failed;
        std::printf("[%s] %s.%s\n", ok ? " ok " : "FAIL", entry.suite.c_str(), entry.name.c_str());
    }
    if (ran == 0) {
        std::fprintf(stderr, "no tests in suite '%s'\n", suite.c_str());
        return 1;
    }
    std::printf("%d tests, %d failed\n", ran, failed);
    return failed == 0 ? 0 : 1;
}
