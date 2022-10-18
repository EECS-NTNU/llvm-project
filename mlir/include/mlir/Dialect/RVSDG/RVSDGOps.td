
include "mlir/Dialect/RVSDG/RVSDGDialect.td"
include "mlir/IR/BuiltinTypes.td"

class RVSDG_Op<string mnemonic, list<Trait> traits = []> :
    Op<RVSDG_Dialect, mnemonic, traits>;

def RVSDG_GammaNode: RVSDG_Op<"gammaNode", [IsolatedFromAbove]> {

    let arguments = (ins
        Index:$predicate,
        Variadic<AnyType>:$inputs
    );

    let results = (outs
        Variadic<AnyType>:$outputs
    );

    let regions = (region
        AnyRegion:$region0,
        AnyRegion:$region1,
        VariadicRegion<AnyRegion>:$additionalRegions
    );

    let assemblyFormat = "attr-dict `(` $predicate `)` custom<TypedParamList>($inputs, type($inputs)) `:` `[` regions `]` `->` type(results)";
}
