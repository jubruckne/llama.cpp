set(TARGET llama-torch-example)

add_executable(${TARGET}
    main.cpp
    torch.cpp
)

install(TARGETS ${TARGET} RUNTIME)

# headers live alongside the sources and rely on the project include directories
# for ggml and gguf dependencies

set(TORCH_EXAMPLE_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml/include
)

target_include_directories(${TARGET} PRIVATE ${TORCH_EXAMPLE_INCLUDES})

target_link_libraries(${TARGET} PRIVATE ggml Threads::Threads)

target_compile_features(${TARGET} PRIVATE cxx_std_23)
