
message("Building RVSDG IR")
add_mlir_dialect(RVSDG rvsdg)
add_mlir_doc(RVSDG RVSDGDialect Dialects/ -gen-dialect-doc)