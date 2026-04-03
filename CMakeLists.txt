cmake_minimum_required(VERSION 3.20.0)
project(ive LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(EXTERNAL_PROJECT_BUILD FALSE)
find_package(MLIR REQUIRED CONFIG)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use mold as the linker if available
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=mold")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=mold")

set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})

# Define the default argument to use by `lit` when testing.
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(IVE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(IVE_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${IVE_SOURCE_DIR}/include)
include_directories(${IVE_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LINK_COMPONENTS
    Core
    Support
    nativecodegen
    OrcJIT
)

set(LLVM_TARGET_DEFINITIONS mlir/IveCombine.td)
mlir_tablegen(mlir/IveCombine.inc -gen-rewriters)
add_public_tablegen_target(IveCombineIncGen)

add_executable(ive)

add_subdirectory(include)
add_subdirectory(lib)

target_sources(ive
    PRIVATE
    main.cpp
    parser/Lexer.cpp
    parser/Parser.cpp
    parser/AST.cpp

    lib/DriverHelpers.cpp

    mlir/Dialect.cpp
    mlir/MLIRGen.cpp    
    mlir/IveCombine.cpp
    mlir/LowerToLLVM.cpp
    mlir/ShapeInferencePass.cpp
    mlir/LowerToAffineLoops.cpp
)

add_dependencies(ive
  IveOpsIncGen
  IveShapeInferenceInterfaceIncGen
  IveCombineIncGen
)

target_link_libraries(ive
    PRIVATE
    MLIRAnalysis
    MLIRBuiltinToLLVMIRTranslation
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRFunctionInterfaces
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRRegisterAllPasses
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIRExport
    MLIRTransforms
)
