"""
agent_strategy.py — 실험 전략 (연산 조합 생성)
2-layer ~ 4-layer 조합 생성
"""
from itertools import product

# 지원 연산 카테고리
CONTRACTION = ["gemm", "batch_gemm", "gemv", "dot_product"]
ELEMENTWISE = ["add", "sub", "mul", "div", "rsqrt", "sqrt",
               "exp", "sigmoid", "tanh", "sin", "cos", "pow2"]

def gen_2layer():
    """contraction → elementwise (48가지)"""
    return [[c, e] for c, e in product(CONTRACTION, ELEMENTWISE)]

def gen_3layer():
    """contraction → elementwise → contraction (192가지)
    batch_gemm은 첫 번째 위치에만 허용 (중간에 오면 shape 불일치)"""
    return [[c1, e, c2] for c1, e, c2 in product(CONTRACTION, ELEMENTWISE, CONTRACTION)
            if c2 != "batch_gemm"]

def gen_4layer():
    """contraction → elementwise → contraction → elementwise (2,304가지)
    batch_gemm은 첫 번째 위치에만 허용"""
    return [[c1, e1, c2, e2]
            for c1, e1, c2, e2 in product(CONTRACTION, ELEMENTWISE, CONTRACTION, ELEMENTWISE)
            if c2 != "batch_gemm"]

def all_combos():
    """2~4 layer 전체 조합, 짧은 것부터 반환"""
    return gen_2layer() + gen_3layer() + gen_4layer()

def combo_id(ops: list) -> str:
    return "-".join(ops)

if __name__ == "__main__":
    two   = gen_2layer()
    three = gen_3layer()
    four  = gen_4layer()
    total = len(two) + len(three) + len(four)
    print(f"2-layer: {len(two):>6,}가지")
    print(f"3-layer: {len(three):>6,}가지")
    print(f"4-layer: {len(four):>6,}가지")
    print(f"전체:    {total:>6,}가지")
    print(f"\n쿨다운 30초 기준 예상 소요 시간:")
    print(f"  CI 평균 3분 + 쿨다운 30초 = 3.5분/실험")
    print(f"  전체: {total * 3.5 / 60:.0f}시간")
    print(f"\n2-layer 예시:")
    for c in two[:3]:
        print(f"  {' → '.join(c)}")
    print(f"\n3-layer 예시:")
    for c in three[:3]:
        print(f"  {' → '.join(c)}")
    print(f"\n4-layer 예시:")
    for c in four[:3]:
        print(f"  {' → '.join(c)}")
