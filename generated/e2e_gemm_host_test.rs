use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_gemm_kernel::{V, M, K, N, pilot_e2e_gemm_kernel};

mod reference_data_e2e_gemm;
use reference_data_e2e_gemm::{reference_a, reference_b, reference_expected};

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let a = HostTensor::<bf16, m![V, M, K]>::from_buf(reference_a());
    let b = HostTensor::<bf16, m![V, K, N]>::from_buf(reference_b());
    let a_hbm = a.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let b_hbm = b.to_hbm(&mut ctx.pdma, 1 << 28).await;
    let _out_hbm = launch(pilot_e2e_gemm_kernel, (&mut ctx, &a_hbm, &b_hbm)).await;
    println!("Pilot E2E gemm: kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// expected는 실제 PyTorch 모델의 입력을 bf16으로 양자화한 뒤 그 값으로
    /// (f32 누산 + bf16 반올림) 계산한 값이다 — RNGD가 실제로 낼 수 있는
    /// "현실적인 정답"이며, Rust가 자체 재계산한 값이 아니다.
    #[tokio::test]
    async fn matches_bf16_reference() {
        let mut ctx = Context::acquire();
        let a = HostTensor::<bf16, m![V, M, K]>::from_buf(reference_a());
        let b = HostTensor::<bf16, m![V, K, N]>::from_buf(reference_b());
        let a_hbm = a.to_hbm(&mut ctx.pdma, 0 << 28).await;
        let b_hbm = b.to_hbm(&mut ctx.pdma, 1 << 28).await;

        let out_hbm = launch(pilot_e2e_gemm_kernel, (&mut ctx, &a_hbm, &b_hbm)).await;
        let actual: Vec<bf16> = out_hbm.to_host::<m![V, M, N]>(&mut ctx.pdma).await.to_buf();
        let expected = reference_expected();

        println!("=== 값 비교 (앞 8개) ===");
        for i in 0..8.min(actual.len()) {
            println!("  [{i}]: {:?} | {:?}", expected[i], actual[i]);
        }

        for (idx, (&e, &av)) in expected.iter().zip(&actual).enumerate() {
            let diff = (f32::from(av) - f32::from(e)).abs();
            let tol = (0.05 * f32::from(e).abs()).max(1.0);
            assert!(diff <= tol, "gemm mismatch at idx={idx}: expected {e:?}, actual {av:?}");
        }
    }
}
