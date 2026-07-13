module attributes {torch.debug_module_name = "DotProduct"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<f32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<f32>
@@HL@@출력 0-초기화 — 대체 후 흡수되어 사라짐 (DCE로 제거)@@SEP@@    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
@@HL@@이 연산 전체가 rngd.dot_product 하나로 대체됨@@SEP@@    %2 = linalg.dot ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%1 : tensor<f32>) -> tensor<f32>
    return %2 : tensor<f32>
  }
}