"builtin.module"() ({
  "ml_program.global"() <{is_mutable, sym_name = "global_seed", sym_visibility = "private", type = tensor<i64>, value = dense<0> : tensor<i64>}> : () -> ()
  "func.func"() <{function_type = (tensor<4x4xf32>) -> tensor<4x4xf32>, sym_name = "forward"}> ({
  ^bb0(%arg0: tensor<4x4xf32>):
    %0 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
    %1 = "tensor.empty"() : () -> tensor<4x4xf32>
    %2 = "rngd.elementwise"(%arg0) {op = "pow2"} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    "func.return"(%2) : (tensor<4x4xf32>) -> ()
  }) : () -> ()
}) {torch.debug_module_name = "UnaryElementwise"} : () -> ()
