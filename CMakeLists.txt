cmake_minimum_required(VERSION 3.21)

project(idol
    VERSION 0.1.0
    DESCRIPTION "Idol image-to-3D inference and splat-to-mesh conversion"
    LANGUAGES CXX)

option(IDOL_BUILD_TOOLS "Build the decoder and mesh command-line tools" ON)
option(IDOL_BUILD_TESTS "Build the unit tests and the Python parity test" ${PROJECT_IS_TOP_LEVEL})
option(IDOL_USE_OPENMP "Run the kernels and the mesh code on several threads with OpenMP" ON)
option(IDOL_HALF_WEIGHTS "Store the packed weights as float16, halving their memory (needs F16C)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(IDOL_TARGET_X86_64 FALSE)
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    # On Apple Silicon the host processor remains arm64 when CMake builds an
    # explicitly selected x86-64 target for Rosetta. Reject mixed targets.
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
        set(IDOL_TARGET_X86_64 TRUE)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(IDOL_TARGET_X86_64 TRUE)
endif()
if(NOT IDOL_TARGET_X86_64)
    message(FATAL_ERROR "idol needs an x86-64 host: its kernels are written for AVX2, FMA and F16C.")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(idol
    src/decoders.cpp
    src/kernels.cpp
    src/mesh.cpp
    src/ply.cpp
    src/weights.cpp)
add_library(idol::idol ALIAS idol)

target_include_directories(idol PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# The kernels call AVX2, FMA and F16C intrinsics directly, so every translation unit that includes them needs the flags.
if(MSVC)
    target_compile_options(idol PUBLIC /arch:AVX2)
    target_compile_options(idol PRIVATE /W4)
else()
    target_compile_options(idol PUBLIC -mavx2 -mfma -mf16c)
    target_compile_options(idol PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(IDOL_HALF_WEIGHTS)
    target_compile_definitions(idol PUBLIC IDOL_HALF_WEIGHTS)
endif()

if(IDOL_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(idol PUBLIC OpenMP::OpenMP_CXX)
endif()

if(IDOL_BUILD_TOOLS)
    add_executable(idol_mesh tools/idol_mesh.cpp)
    add_executable(idol_decode tools/idol_decode.cpp)
    target_include_directories(idol_decode PRIVATE src)
    add_executable(idol_octree_compare tools/idol_octree_compare.cpp)
    target_include_directories(idol_octree_compare PRIVATE src)
    target_link_libraries(idol_decode PRIVATE idol::idol)
    target_link_libraries(idol_octree_compare PRIVATE idol::idol)
    if(WIN32 AND NOT MSVC)
        target_link_libraries(idol_decode PRIVATE psapi)
    endif()
    target_link_libraries(idol_mesh PRIVATE idol::idol)
endif()

if(IDOL_BUILD_TESTS)
    enable_testing()

    add_executable(idol_tests
        tests/test_main.cpp
        tests/test_kernels.cpp
        tests/test_mesh.cpp
        tests/test_ply.cpp
        tests/test_octree_compare.cpp
        tests/test_flux2_tokens.cpp)
    target_include_directories(idol_tests PRIVATE src tests)
    target_link_libraries(idol_tests PRIVATE idol::idol)

    add_test(NAME kernels COMMAND idol_tests kernels)
    add_test(NAME mesh COMMAND idol_tests mesh)
    add_test(NAME ply COMMAND idol_tests ply)
    add_test(NAME octree_compare COMMAND idol_tests octree_compare)
    add_test(NAME flux2_tokens COMMAND idol_tests flux2_tokens)

    # The parity test needs PyTorch and the export script. It reports "skipped" (exit 77) when torch is missing.
    # Float16 weights are less precise, so the build allows a larger difference from PyTorch.
    if(IDOL_HALF_WEIGHTS)
        set(IDOL_PARITY_TOLERANCE 1e-2)
    else()
        set(IDOL_PARITY_TOLERANCE 2e-6)
    endif()
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_executable(idol_flux2_tokens_probe tests/flux2_tokens_probe.cpp)
        target_include_directories(idol_flux2_tokens_probe PRIVATE src)
        add_test(NAME flux2_tokens_parity COMMAND Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_flux2_tokens_parity.py"
            "$<TARGET_FILE:idol_flux2_tokens_probe>")
        set_tests_properties(flux2_tokens_parity PROPERTIES SKIP_RETURN_CODE 77)

        add_test(NAME quant_methods COMMAND Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_quant_methods.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools")
        set_tests_properties(quant_methods PROPERTIES SKIP_RETURN_CODE 77)

        add_test(NAME full_gguf COMMAND Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_full_gguf.py")
        set_tests_properties(full_gguf PROPERTIES SKIP_RETURN_CODE 77)

        add_test(NAME fused_packed_gpu COMMAND Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_fused_packed.py")
        set_tests_properties(fused_packed_gpu PROPERTIES SKIP_RETURN_CODE 77)

        add_test(NAME calibrated_gguf COMMAND Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_calibrated_gguf.py")
        set_tests_properties(calibrated_gguf PROPERTIES SKIP_RETURN_CODE 77)

        add_test(NAME dino_export COMMAND Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_dino_export.py")
        set_tests_properties(dino_export PROPERTIES SKIP_RETURN_CODE 77)

        add_test(NAME compare_splats COMMAND Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_compare_splats.py")
        set_tests_properties(compare_splats PROPERTIES SKIP_RETURN_CODE 77)

        add_test(NAME idol_session COMMAND Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_idol_session.py")
        set_tests_properties(idol_session PROPERTIES SKIP_RETURN_CODE 77)

        add_test(NAME resident_session_routing COMMAND Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_resident_session_routing.py")

        add_test(NAME idol_staging COMMAND Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_idol_staging.py")
        set_tests_properties(idol_staging PROPERTIES SKIP_RETURN_CODE 77)

        add_test(NAME render_splats_preview COMMAND Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_render_splats_preview.py")
        set_tests_properties(render_splats_preview PROPERTIES SKIP_RETURN_CODE 77)

        add_test(NAME compare_meshes COMMAND Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_compare_meshes.py")
        set_tests_properties(compare_meshes PROPERTIES SKIP_RETURN_CODE 77)

        add_test(NAME calibration_preflight COMMAND Python3::Interpreter
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_calibration_preflight.py")
        set_tests_properties(calibration_preflight PROPERTIES SKIP_RETURN_CODE 77)

        # The real Idol decoders against the official SDK, on small random ones. Needs the SDK source in IDOL_SDK.
        if(TARGET idol_decode)
            add_executable(idol_decoder_probe tests/decoder_probe.cpp)
            target_link_libraries(idol_decoder_probe PRIVATE idol::idol)
            add_test(NAME decoders COMMAND Python3::Interpreter
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_decoders.py"
                "$<TARGET_FILE:idol_decode>"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools"
                "${IDOL_PARITY_TOLERANCE}"
                "$<TARGET_FILE:idol_decoder_probe>"
                "$<TARGET_FILE:idol_octree_compare>")
            set_tests_properties(decoders PROPERTIES SKIP_RETURN_CODE 77)
            add_test(NAME gguf_decoder COMMAND Python3::Interpreter
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_gguf_decoder.py"
                "$<TARGET_FILE:idol_decode>")
            set_tests_properties(gguf_decoder PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endif()
endif()

if(IDOL_BUILD_TOOLS)
    install(TARGETS idol_mesh idol_decode idol_octree_compare RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(TARGETS idol
    EXPORT idolTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT idolTargets
    NAMESPACE idol::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/idol)

configure_package_config_file(
    cmake/idolConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/idolConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/idol)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/idolConfigVersion.cmake
    COMPATIBILITY SameMinorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/idolConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/idolConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/idol)
