"builtin.module"() ({
  "ml_program.global"() <{is_mutable, sym_name = "global_seed", sym_visibility = "private", type = tensor<i64>, value = dense<0> : tensor<i64>}> : () -> ()
  "func.func"() <{function_type = (tensor<32x32x32xf32>, tensor<32x32x8xf32>) -> tensor<32x32x8xf32>, sym_name = "forward"}> ({
  ^bb0(%arg0: tensor<32x32x32xf32>, %arg1: tensor<32x32x8xf32>):
@@HL@@Contraction Engine(TRF + contract_outer/packet/time/lane)이 실제 계산 수행@@SEP@@    %0 = "rngd.batch_gemm"(%arg0, %arg1) : (tensor<32x32x32xf32>, tensor<32x32x8xf32>) -> tensor<32x32x8xf32>
    "func.return"(%0) : (tensor<32x32x8xf32>) -> ()
  }) : () -> ()
}) {torch.debug_module_name = "BatchMatmul"} : () -> ()