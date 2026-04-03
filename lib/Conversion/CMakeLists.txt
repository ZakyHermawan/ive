set(LLVM_TARGET_DEFINITIONS IveToSCF.td)
mlir_tablegen(IveToSCF.hpp.inc -gen-pass-decls)
add_public_tablegen_target(IveToSCFGen)

target_include_directories(ive PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

target_sources(ive
	PRIVATE
	IveToSCF.cpp
	LowerToAffineLoops.cpp
	LowerToLLVM.cpp
)

add_dependencies(ive
	IveToSCFGen
)
