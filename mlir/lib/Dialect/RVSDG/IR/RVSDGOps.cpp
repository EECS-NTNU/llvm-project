#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

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
    p.printOperand(operands[i]);
    p << ": ";
    p.printType(types[i]);
  }
  p << ")";
}

ParseResult
parseTypedParamList(OpAsmParser &parser,
                    SmallVectorImpl<OpAsmParser::OperandType> &operands,
                    SmallVectorImpl<Type> &types) {

  if (parser.parseLParen().failed()) {
    return ParseResult::failure();
  }
  unsigned int index = 0;
  while (parser.parseOptionalRParen().failed()) {
    if (index != 0) {
      if (parser.parseComma().failed()) {
        return ParseResult::failure();
      }
    }
    mlir::OpAsmParser::OperandType operand;
    if (parser.parseOperand(operand).failed()) {
      return ParseResult::failure();
    }
    Type type;
    if (parser.parseColonType(type).failed()) {
      return ParseResult::failure();
    }
    operands.push_back(operand);
    types.push_back(type);
    ++index;
  }

  return ParseResult::success();
}

LogicalResult GammaOutput::verify() {
  auto parent = cast<GammaNode>((*this)->getParentOp());
  const auto &results = parent.getResults();
  if (getNumOperands() != results.size()) {
    return emitOpError("has ")
           << getNumOperands() << " operands, but parent node outputs "
           << results.size();
  }

  for (unsigned i = 0; i < results.size(); ++i) {
    if (getOperand(i).getType() != results[i].getType()) {
      return emitError() << "type of output operand " << i << " ("
                         << getOperand(i).getType()
                         << ") does not match node output type ("
                         << results[i].getType() << ")";
    }
  }

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/RVSDG/Ops.cpp.inc"