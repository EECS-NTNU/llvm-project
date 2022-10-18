
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

#include "mlir/Dialect/RVSDG/Dialect.h.inc"
#define GET_OP_CLASSES
#include "mlir/Dialect/RVSDG/Ops.h.inc"


void mlir::rvsdg::RVSDGDialect::initialize(void){
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/RVSDG/Ops.cpp.inc"
    >();
}

#include "mlir/Dialect/RVSDG/Dialect.cpp.inc"