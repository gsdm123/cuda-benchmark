# set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 80)

# add verification option
option(ENABLE_VERIFICATION "Enable result verification in benchmarks" OFF)

if(ENABLE_VERIFICATION)
    add_compile_definitions(ENABLE_VERIFICATION)
endif()

# add benchmark executable
add_executable(run_benchmark
    basic_benchmark.cu
    runtime_benchmark.cu
    memory_benchmark.cu
    stream_benchmark.cu
    unified_memory_benchmark.cu
    advanced_benchmark.cu
    error_benchmark.cu
    interaction_benchmark.cu
    config_benchmark.cu
    mma_benchmark.cu
    conv_benchmark.cu
)

# link Google Benchmark
target_link_libraries(run_benchmark
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    CUDA::cudart
)

# set compile options
target_compile_features(run_benchmark PRIVATE cxx_std_17)
set_target_properties(run_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# add include directory
target_include_directories(run_benchmark
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
