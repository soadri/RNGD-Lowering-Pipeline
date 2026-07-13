"builtin.module"() ({
  "ml_program.global"() <{is_mutable, sym_name = "global_seed", sym_visibility = "private", type = tensor<i64>, value = dense<0> : tensor<i64>}> : () -> ()
  "func.func"() <{function_type = (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>, sym_name = "forward"}> ({
  ^bb0(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>):
    %0 = "tensor.empty"() : () -> tensor<4x4xf32>
@@HL@@실제 연산자 — attribute op="mul"에 연산 종류가 담김@@SEP@@    %1 = "rngd.elementwise"(%arg0, %arg1) {op = "mul"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    "func.return"(%1) : (tensor<4x4xf32>) -> ()
  }) : () -> ()
}) {torch.debug_module_name = "ElementwiseBinary"} : () -> ()