# Current Experiment Snapshot

This file is a plain snapshot of the current experiment templates and result artifacts in `project/`.
It is intended for restoring the manuscript after pulling back the original source files.

## Result Availability

Three case studies:

- `ex1`: one-dimensional Kuramoto oscillator
- `ex2`: two-dimensional Kuramoto oscillator
- `ex3`: two-room temperature system

Result status by method:

| Case | PT | NNT | CC |
| --- | --- | --- | --- |
| `ex1` | Success | Success | Success |
| `ex2` | Success | Success | Timeout |
| `ex3` | Success | Success | Timeout |

Source result files:

- `project/src/ex1/PT/res_pt_ex1.json`
- `project/src/ex2/PT/res_pt_ex2.json`
- `project/src/ex3/PT/res_pt_ex3.json`
- `project/src/ex1/NNT/res_nnt_ex1.json`
- `project/src/ex2/NNT/res_nnt_ex2.json`
- `project/src/ex3/NNT/res_nnt_ex3.json`
- `project/src/ex1/CC/res_cc_ex1.json`
- `project/src/ex2/CC/res_cc_ex2.json`
- `project/src/ex3/CC/res_cc_ex3.json`

## PT Templates

### ex1 PT

Source: `project/src/ex1/PT/main.py`

Template:

```text
B(x, q) = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*q + c6*q^2 + c7*q^3 + c8*x*q
```

Result:

```json
{
  "success": true,
  "coefficients": [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, -0.4375],
  "elapsed_sec": 1.7593903541564941,
  "certificate_type": "transition_safety"
}
```

### ex2 PT

Source: `project/src/ex2/PT/main.py`

Template:

```text
B(x1, x2, q) = c0 + c1*x1 + c2*x2 + c3*q + c4*I_Xu(x) + c5*x1*q + c6*x2*q
```

Result:

```json
{
  "success": true,
  "coefficients": [-1.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0],
  "elapsed_sec": 1.7244622707366943,
  "certificate_type": "transition_safety",
  "target_q": 0
}
```

### ex3 PT

Source: `project/src/ex3/PT/main.py`

Template:

```text
B(x1, x2) = c0
          + c1*I_X0(x1,x2)
          + c2*I_VF(x1,x2)
          + c3*x1*I_X0(x1,x2)
          + c4*x2*I_X0(x1,x2)
          + c5*x1*I_VF(x1,x2)
          + c6*x2*I_VF(x1,x2)
          + c7*max(x1,x2)
          + c8*x1^2
          + c9*x2^2
```

Result:

```json
{
  "success": true,
  "epsilon": 0.25,
  "coefficients": [0.0, 0.0, 27.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
  "elapsed_sec": 2.676501750946045,
  "certificate_type": "transition_persistence"
}
```

## NNT Templates And Parameters

### ex1 NNT

Source:

- `project/src/ex1/NNT/main.py`
- `project/src/ex1/NNT/res_nnt_ex1.json`
- `project/saved_models/barrier_net_20260408_013924/barrier_net.pth`

Template:

```text
B(x, q) = fc2(ReLU(fc1([x, q])))
Architecture: 2 -> 10 -> 1
```

Result:

```json
{
  "success": true,
  "iterations": 1,
  "elapsed_sec": 6.807429075241089,
  "hidden_dim": 10,
  "model_state_path": "saved_models/barrier_net_20260408_013924/barrier_net.pth",
  "certificate_type": "transition_safety"
}
```

Extracted parameters:

```json
{
  "fc1.weight": [[0.3542354106903076, 0.03017079085111618], [0.4034789204597473, 0.1548570990562439], [-0.33290594816207886, 0.20670443773269653], [0.059538308531045914, -0.5708100199699402], [-0.7776458263397217, -0.30270880460739136], [0.2816714942455292, -0.5657135248184204], [-0.00433345464989543, 0.27621468901634216], [-0.11347808688879013, -0.37141919136047363], [0.3017216622829437, -0.1451742947101593], [0.0013668136671185493, 0.3228271007537842]],
  "fc1.bias": [-0.8349039554595947, 0.5795674324035645, -0.47568362951278687, -0.27932465076446533, -0.2889058291912079, 0.16922584176063538, -0.6695564389228821, -0.23689666390419006, 0.49750402569770813, 0.5533573627471924],
  "fc2.weight": [[-1.3667676448822021, 0.13470453023910522, -0.015854772180318832, -0.04601728171110153, 0.22054167091846466, -0.15794339776039124, -0.09915264695882797, 0.003246989334002137, -0.02935893088579178, 0.1876564770936966]],
  "fc2.bias": [-0.252432256937027]
}
```

### ex2 NNT

Source:

- `project/src/ex2/NNT/main.py`
- `project/src/ex2/NNT/res_nnt_ex2.json`
- `project/saved_models/barrier_net_ex2_20260408_013944/parameters.json`

Template:

```text
B(x1, x2, q) = fc2(ReLU(fc1([x1, x2, q])))
Architecture: 3 -> 15 -> 1
```

Result:

```json
{
  "success": true,
  "iterations": 11,
  "elapsed_sec": 110.9514217376709,
  "hidden_dim": 15,
  "model_state_path": "saved_models/barrier_net_ex2_20260408_013944/barrier_net.pth",
  "certificate_type": "transition_safety"
}
```

Extracted parameters:

```json
{
  "W1": [[-0.017978008836507797, 0.6406944394111633, -0.03856681287288666], [-0.5729965567588806, -0.14924462139606476, -0.043293245136737823], [-0.5660426616668701, -0.2565310001373291, 0.10918396711349487], [-0.12088224291801453, 0.6219276785850525, -0.9516218304634094], [-0.5492688417434692, -0.4527278244495392, 0.2669191062450409], [-0.3572467267513275, 0.10176215320825577, -0.2604440450668335], [-0.6215685606002808, 0.4530791640281677, -0.636301577091217], [-0.5046539902687073, -0.0016143012326210737, 0.04746519774198532], [-0.21175090968608856, 0.12297604978084564, -0.5237511992454529], [0.6063219308853149, 0.14016705751419067, -2.3412678241729736], [-0.0786895900964737, -0.15929390490055084, -0.10671024024486542], [0.6868988275527954, -0.5882328748703003, -0.651290774345398], [-0.7519702315330505, 0.13900414109230042, -0.018297474831342697], [-0.551321804523468, 0.1751272827386856, -0.04693244770169258], [-0.18292713165283203, 0.10734313726425171, -0.20697620511054993]],
  "b1": [-0.23332440853118896, -0.2526431083679199, -0.20304657518863678, 0.004985474515706301, -0.4151919186115265, -0.18106091022491455, -0.32431474328041077, 0.8322128057479858, -0.5451726317405701, 0.0707450658082962, -0.18164202570915222, 0.44940027594566345, 0.6089276075363159, -0.6383590698242188, 0.3688134551048279],
  "W2": [[-0.2210632711648941, 0.12717370688915253, -0.049974024295806885, -0.5093038082122803, -0.1825152486562729, 0.08938224613666534, 0.4823448359966278, -0.43327656388282776, -0.12058054655790329, 0.4247071146965027, 0.11804953217506409, -0.6537949442863464, 0.25955432653427124, 0.05265373736619949, 0.011966556310653687]],
  "b2": [0.3641239106655121]
}
```

### ex3 NNT

Source:

- `project/src/ex3/NNT/main.py`
- `project/src/ex3/NNT/res_nnt_ex3.json`
- `project/persistence_barrier_model_2layer.pth`

Template:

```text
B(x1, x2) = fc2((fc1([x1, x2]))^2)
Architecture: 2 -> 3 -> 1
Output layer has no bias
epsilon = 0.01
```

Result:

```json
{
  "success": true,
  "elapsed_sec": 1461.9093821048737,
  "hidden_dim": 3,
  "epsilon": 0.01,
  "model_state_path": "persistence_barrier_model_2layer.pth",
  "certificate_type": "transition_persistence"
}
```

Extracted parameters:

```json
{
  "fc1.weight": [[-0.12892259657382965, 0.07935464382171631], [0.3997109830379486, -0.35251060128211975], [-0.6177504062652588, 0.6178852915763855]],
  "fc1.bias": [1.4433966875076294, -1.4130280017852783, -0.016157710924744606],
  "fc2.weight": [[1.2257390022277832, 1.889444351196289, 1.192406177520752]]
}
```

## CC Results

### ex1 CC

Source: `project/src/ex1/CC/res_cc_ex1.json`

```json
{
  "success": true,
  "epsilon": 0.5,
  "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.20462740678906, 4.6260833e-07, 0.20462777053779, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "elapsed_sec": 0.36569881439208984
}
```

### ex2 CC

Source: `project/src/ex2/CC/res_cc_ex2.json`

```json
{
  "success": false,
  "iterations": 60,
  "elapsed_sec": 0.32466578483581543
}
```

Interpretation: `Timeout`

### ex3 CC

Source: `project/src/ex3/CC/res_cc_ex3.json`

```json
{
  "success": false,
  "mode": "closure",
  "iterations": 1,
  "elapsed_sec": 58.76244330406189
}
```

Interpretation: `Timeout`

## Restore Note

If you pull back the original manuscript files, the minimum result statements to restore are:

- PT: all three cases have results
- NNT: all three cases have results
- CC: only `ex1` has a result
- CC for `ex2` and `ex3`: write `Timeout`
