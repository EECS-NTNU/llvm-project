#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

#include "mlir/Dialect/RVSDG/Dialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/RVSDG/Ops.h.inc"

#include <stdio.h>

using namespace mlir;
using namespace rvsdg;

/**
 * @brief Prints out a comma separated list of parameters paired with their
 * respective types. Having types as a parameter is redundant, but tablegen
 * won't build without it.
 *
 * @param p Assembly printer
 * @param op Operation which we are printing
 * @param operands Range of operands to be printed
 * @param types Types of the operands will be matched with operands using
 *              position in the array
 **/
void printTypedParamList(OpAsmPrinter &p, Operation *op, OperandRange operands,
                         TypeRange types) {
  
  p << "(";
  int param_count = std::min(operands.size(), types.size());
  for (int i = 0; i < param_count; ++i) {
    if (i != 0) {
      p << ", ";
    }
    p.printType(types[i]);
    p << " ";
    p.printOperand(operands[i]);
  }
  p << ")";
}

ParseResult
parseTypedParamList(OpAsmParser &parser,
                    SmallVectorImpl<OpAsmParser::OperandType> &operands,
                    SmallVectorImpl<Type> &types) {

  auto parseTypedParam = [&]() -> ParseResult {
    Type result;
    if (parser.parseType(result).succeeded()) {
      return ParseResult::failure();
    };
    OpAsmParser::OperandType operand_res;
    if (parser.parseOperand(operand_res)) {
      return ParseResult::failure();
    }
    return ParseResult::success();
  };

  parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                 parseTypedParam);
  return ParseResult::success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/RVSDG/Ops.cpp.inc"