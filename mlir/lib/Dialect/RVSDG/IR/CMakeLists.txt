add_mlir_dialect_library(MLIRRVSDG
    RVSDGDialect.cpp
    RVSDGOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/RVSDG
)