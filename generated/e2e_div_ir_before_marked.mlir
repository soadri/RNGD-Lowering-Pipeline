#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "ElementwiseBinary"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
@@HL@@출력 버퍼 초기화 — 대체 후 불필요해져 제거됨@@SEP@@    %0 = tensor.empty() : tensor<4x4xf32>
@@HL@@이 블록 전체가 rngd.elementwise 하나로 대체됨@@SEP@@    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%0 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
@@HL@@연산 종류 결정 → rngd.elementwise attribute op="div"@@SEP@@      %2 = arith.divf %in, %in_0 : f32
@@HL@@generic 내부 결과 반환 — 대체 후 제거됨@@SEP@@      linalg.yield %2 : f32
    } -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}