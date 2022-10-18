// RUN: mlir-opt %s | mlir-opt | FileCheck %s

module {
    // CHECK: [[PREDICATE:%.+]] = arith.constant 0 : index
    // CHECK: rvsdg.gammaNode
    // CHECK-SAME = ([[PREDICATE]]) 
    // CHECK-SAME () : [{
    // CHECK-NEXT: }, {
    // CHECK-NEXT: }] -> i32
    
    %predicate = arith.constant 0 : index
    %0 = rvsdg.gammaNode (%predicate) () : [{}, {}] -> i32
}
