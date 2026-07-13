"builtin.module"() ({
  "ml_program.global"() <{is_mutable, sym_name = "global_seed", sym_visibility = "private", type = tensor<i64>, value = dense<0> : tensor<i64>}> : () -> ()
  "func.func"() <{function_type = (tensor<32xf32>, tensor<32xf32>) -> tensor<f32>, sym_name = "forward"}> ({
  ^bb0(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>):
    %0 = "rngd.dot_product"(%arg0, %arg1) : (tensor<32xf32>, tensor<32xf32>) -> tensor<f32>
    "func.return"(%0) : (tensor<f32>) -> ()
  }) : () -> ()
}) {torch.debug_module_name = "DotProduct"} : () -> ()
