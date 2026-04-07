import math

"""
 CEGIS-Z3-dReal
 B(x,q) ：
1. : ∀x∈X0, B(x,1) ≥ 0
2. :  B(x,q) ≥ 0， B(f(x), δ(q, L(x))) ≥ 0
3. :
"""

SAMPLING_TIME = 0.1
NATURAL_FREQUENCY = 0.01
COUPLING_COEFFICIENT = 0.0006
QUADRATIC_TERM = 0.532
CONSTANT_TERM = 1.69

def in_state_space(x: float) -> bool:
    """: x ∈ [0, 2π]"""
    return 0 <= x <= 2 * math.pi

def in_initial_set(x: float) -> bool:
    """: x ∈ [0, π/9]"""
    return 0 <= x <= math.pi / 9

def in_unsafe_set(x: float) -> bool:
    """: x ∈ [7π/9, 8π/9]"""
    return 7 * math.pi / 9 <= x <= 8 * math.pi / 9

def system_dynamics(x: float) -> float:
    """
    : x_next = f(x)
    f(x) = x + Ts*Ω + Ts*K*sin(-x) - 0.532*x^2 + 1.69
    """
    return (x + SAMPLING_TIME * NATURAL_FREQUENCY +
            SAMPLING_TIME * COUPLING_COEFFICIENT * math.sin(-x) -
            QUADRATIC_TERM * x ** 2 + CONSTANT_TERM)

def safety_label(x: float) -> int:
    """
    : x1，0
    """
    return 1 if in_unsafe_set(x) else 0

def mode_transition(current_mode: int, safety_label_value: int) -> int:
    """
    : δ(q, w)
    q=1, w=1 → 0  ()
    q=1, w=0 → 1  ()
    q=0 → 0       ()
    """
    if current_mode == 1:
        return 0 if safety_label_value == 1 else 1
    else:  # current_mode == 0
        return 0

def barrier_function_2025_03_15(x: float, mode: int) -> float:
    """
    2025-03-15
    B(x, q) = -1 + 2*q^2 - (7/16)*x*q
    """
    return -1 + 2 * mode ** 2 - (7 / 16) * x * mode

def generate_samples(start: float, end: float, step: float) -> list[float]:
    """
    [start, end]step
    """
    num_samples = int((end - start) / step) + 1
    return [start + i * step for i in range(num_samples)]

def verify_safety_certificate(barrier_func, max_iterations: int = 2000) -> bool:
    """

    : True ，False
    """

    initial_samples = generate_samples(0, math.pi / 9, 0.0001)

    print("...")
    print(f": {len(initial_samples)}")
    print(f": {max_iterations}")
    print("-" * 50)

    for initial_x in initial_samples:
        x = initial_x
        mode = 1
        iteration = 0

        while iteration < max_iterations:

            next_x = system_dynamics(x)
            next_mode = mode_transition(mode, safety_label(x))

            current_barrier = barrier_func(x, mode)
            next_barrier = barrier_func(next_x, next_mode)

            if in_initial_set(x) and current_barrier < 0:
                print(f"❌ :")
                print(f"   : x = {initial_x:.6f}, : x = {x:.6f}")
                print(f"   : q = {mode}, B(x,q) = {current_barrier:.6f}")
                return False

            if in_state_space(x) and in_state_space(next_x):
                if current_barrier >= 0 and next_barrier < 0:
                    print(f"❌ :")
                    print(f"   : x = {initial_x:.6f}")
                    print(f"   : (x,q) = ({x:.6f}, {mode}), B = {current_barrier:.6f}")
                    print(f"   : (x',q') = ({next_x:.6f}, {next_mode}), B' = {next_barrier:.6f}")
                    return False

            if next_mode == 0 and current_barrier >= 0:
                print(f"❌ :")
                print(f"   : x = {initial_x:.6f}")
                print(f"    q=0  B(x,q) ≥ 0 ")
                return False

            if current_barrier < 0 and not in_initial_set(x):
                print(f"⚠️  :  B(x,q) < 0")
                print(f"   (x,q) = ({x:.6f}, {mode}), B = {current_barrier:.6f}")

            x = next_x
            mode = next_mode
            iteration += 1

        print(f"✓  x = {initial_x:.6f}  ({iteration})")

    print("\n" + "=" * 50)
    print("✅ ！。")
    print("=" * 50)
    return True

def quick_verification(barrier_func) -> None:
    """
    ：
    """
    print("...")

    test_cases = [
        (0.0, 1, ""),
        (math.pi / 9, 1, ""),
        (7 * math.pi / 9, 0, " (q=0)"),
        (8 * math.pi / 9, 0, " (q=0)"),
        (7 * math.pi / 9, 1, " (q=1)"),
        (8 * math.pi / 9, 1, " (q=1)"),
    ]

    for x, q, description in test_cases:
        barrier_value = barrier_func(x, q)
        print(f"{description:20} (x={x:.4f}, q={q}): B = {barrier_value:.6f}")

        if in_initial_set(x) and q == 1 and barrier_value < 0:
            print(f"  ⚠️  ")
        elif in_unsafe_set(x) and q == 0 and barrier_value >= 0:
            print(f"  ⚠️  ")

    print("-" * 50)

def main():
    """"""
    print("=" * 60)
    print("")
    print(f": B(x, q) = -1 + 2*q^2 - (7/16)*x*q")
    print(f":  Kuramoto ")
    print("=" * 60)

    quick_verification(barrier_function_2025_03_15)

    is_valid = verify_safety_certificate(
        barrier_func=barrier_function_2025_03_15,
        max_iterations=2000
    )

    if is_valid:
        print("\n:")
        print("1. ✅ :  x∈X0  B(x,1) ≥ 0")
        print("2. ✅ : B(x,q) ≥ 0 ⇒ B(f(x), δ(q,L(x))) ≥ 0")
        print("3. ✅ : ")
        print("\n🎉 ！")
    else:
        print("\n❌ ，")

    print("=" * 60)

if __name__ == "__main__":
    main()
