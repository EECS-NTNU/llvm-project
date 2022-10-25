#pragma once

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace rvsdg {

/**
 * Check if an operand range contains a specific operand.
 * Utility method for PassThroughOperands trait
 */
static bool operandRangeContains(OperandRange &range, Value &operand) {
  for (Value const &rOperand : range) {
    if (operand == rOperand) {
      return true;
    }
  }
  return false;
}

/**
 * Verifies that all used values are either defined within the regions
 * of the operation that implements the trait or have been explicitly
 * listed in the operands of the implementing operation.
 */
template <typename ConcreteType>
class PassThroughOperands
    : public TypeTrait::TraitBase<ConcreteType, PassThroughOperands> {
public:
  /**
   * Verifies trait properties
   */
  static LogicalResult verifyTrait(Operation *passThroughOp) {
    assert(passThroughOp->hasTrait<PassThroughOperands>() &&
           "Intended to check PassThroughOperands ops");

    // Operands of implementing operation
    OperandRange passThroughOperands = passThroughOp->getOperands();

    // Perform depth-first traversal of all operands
    SmallVector<Region *, 8> pendingRegions;
    for (Region &region : passThroughOp->getRegions()) {
      pendingRegions.push_back(&region);

      while (!pendingRegions.empty()) {
        for (Operation &op : pendingRegions.pop_back_val()->getOps()) {
          for (Value operand : op.getOperands()) {

            // Should not trigger for well formed IR
            if (!operand)
              return op.emitOpError("operation's operand is null");

            // Region where the value is defined
            auto *operandRegion = operand.getParentRegion();
            if (!region.isAncestor(operandRegion) &&
                !operandRangeContains(passThroughOperands, operand)) {
              return op.emitOpError("using value defined outside the region which is not passed through in the operands");
            }
          }

          if (op.getNumRegions() > 0 && !op.hasTrait<PassThroughOperands>()) {
            for (Region &subRegion : op.getRegions()) {
              pendingRegions.push_back(&subRegion);
            }
          }
        }
      }
    }
    return success();
  }
};

} // namespace rvsdg
} // namespace mlir
