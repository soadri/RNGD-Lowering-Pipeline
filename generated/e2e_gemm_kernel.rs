use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.gemm — confirmed (pilot_batch_gemm 구조 재사용)
// 배치 축이 없는 2D matmul(gemm)은 V=1인 batch_gemm으로 취급 (동일 검증 구조 재사용)
axes![V = 1, M = 32, K = 32, N = 8];

pub type Chip = m![1];
pub type Cluster = m![1];
// V<=256 가정: 하드웨어 제약(Slice ∈ {64,128,192,256})에 맞춰 256-slice 공간에 매핑
pub type Slice = m![V # 256];
// contract_lane의 OutPacket::SIZE는 정확히 8이어야 함(실측 확인) -> N<8이면 8-lane 공간에 패딩
pub type Lane = m![N # 8];

#[device(chip = 1)]
pub fn pilot_e2e_gemm_kernel(
    ctx: &mut Context,
    a: &HbmTensor<bf16, Chip, m![V, M, K]>,
    b: &HbmTensor<bf16, Chip, m![V, K, N]>,
) -> HbmTensor<bf16, Chip, m![V, M, N]> {
    let a: DmTensor<bf16, Chip, Cluster, Slice, m![M, K]> = a.to_dm(&mut ctx.tdma, 0);
    let b: DmTensor<bf16, Chip, Cluster, Slice, m![K, N]> = b.to_dm(&mut ctx.tdma, 1 << 12);

    let b_trf: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![K]> = ctx
        .sub
        .begin(b.view())
        .fetch::<m![N], m![K]>()
        .collect::<m![N, K / 16], m![K % 16]>()
        .to_trf(TrfAddress::Full);

    let result: DmTensor<bf16, Chip, Cluster, Slice, m![M, N]> = ctx
        .main
        .begin(a.view())
        .fetch::<m![M], m![K]>()
        .collect::<m![M, K / 16], m![K % 16]>()
        .contract_outer::<m![M], m![K], _, _>(&b_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![M]>()
        .contract_lane::<m![M], m![N # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![N # 16]>()
        .commit_trim::<m![N]>()
        .commit(0);

    result.to_hbm(&mut ctx.tdma, 2 << 28)
}
