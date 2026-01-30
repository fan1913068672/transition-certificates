import omega.automata
from omega import logic as lo


# 定义 LTL 公式
ltl_formula = 'G F a'

# 转换为 Büchi 自动机
automaton = omega.automata.TransitionSystem(ltl_formula)

# 打印自动机信息
print("初始状态:", automaton.initial)
print("接受条件:", automaton.accepting)
print("状态转移:")
for (src, label), dst in automaton.transitions.items():
    print(f"  状态 {src} 在条件 {label} 下转移到 {dst}")