#ifndef RVSDG_TRAITS
#define RVSDG_TRAITS

def PassThroughOperands : NativeOpTrait<"PassThroughOperands">{
  let cppNamespace = "::mlir::rvsdg";
}


#endif //RVSDG_TRAITS