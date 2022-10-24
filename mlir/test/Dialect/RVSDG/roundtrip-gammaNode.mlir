// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// COM: Round-trip of an empty gamma node
// CHECK-LABEL: rvsdg_empty_gamma_node
func @rvsdg_empty_gamma_node() -> (index){
    // CHECK: [[PREDICATE:%.+]]
    // CHECK: rvsdg.gammaNode
    // CHECK-SAME = ([[PREDICATE]]) 
    // CHECK-SAME () : [{
    // CHECK-NEXT: }, {
    // CHECK-NEXT: }] -> i32
    
    %predicate = arith.constant 0 : index
    %0 = rvsdg.gammaNode (%predicate) () : [{}, {}] -> i32
    return %predicate :index
}
