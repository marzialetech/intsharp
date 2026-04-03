# Interface Sharpening RHS Grammar

This document defines the **allowed building blocks** for proposing new
interface sharpening source terms. Any AI-proposed or human-designed RHS
must be expressible using only these primitives.

## 1. Scalar Fields

| Symbol | Definition | Notes |
|--------|-----------|-------|
| `psi` | Volume fraction ψ ∈ [0,1] | Primary unknown |
| `1-psi` | Complement | |
| `psi*(1-psi)` | Compressive weight | Peaks at interface (ψ=0.5) |
| `phi_inv` | ε ln(ψ/(1−ψ)) | Algebraic signed distance |
| `phi_map(alpha)` | ψ^α / (ψ^α + (1−ψ)^α) | Mapped phase field |

## 2. Gradient and Divergence Operators

| Operator | Meaning | Notes |
|----------|---------|-------|
| `grad(f)` | Central difference ∇f | 1D: (f_{i+1}−f_{i−1})/(2dx) |
| `div(F)` | Divergence of flux F | Central or Rusanov face-flux |
| `laplacian(f)` | ∇²f = (f_{i+1}−2f_i+f_{i−1})/dx² | Second-order |
| `abs_grad(f)` | \|∇f\| | Gradient magnitude |

## 3. Normal Vectors

| Symbol | Definition | Used by |
|--------|-----------|---------|
| `n_psi` | grad(ψ)/\|grad(ψ)\| | CL, OK, LCLS |
| `n_phi` | grad(φ_inv)/\|grad(φ_inv)\| | ACLS, CLS 2017 |
| `n_map` | grad(φ_map)/\|grad(φ_map)\| | CLS 2010, 2015 |
| `m_scls` | ε∇ψ / √(ε²\|∇ψ\|² + α²e^{−β ε²\|∇ψ\|²}) | SCLS |

## 4. Localization Weights

| Weight w(ψ) | Expression | Used by |
|-------------|-----------|---------|
| `1` | No localization | CL, OK, ACLS |
| `4*psi*(1-psi)` | β = 4ψ(1−ψ) | LCLS 2012, 2014 |
| `1/(4*cosh²(φ/(2ε)))` | ≈ ψ(1−ψ) | CLS 2017 |
| Custom `w(psi, params)` | User-defined | New proposals |

## 5. Flux Templates

### Conservative (divergence form)

```
∂ψ/∂τ + div(F) = 0
```

Standard CLS flux:

```
F = ε * D(ψ, n) − C(ψ) * n
```

where:
- `D(ψ, n)` is the **diffusion** term:
  - Isotropic: `grad(psi)` (CL)
  - Anisotropic: `(grad(psi) · n) * n` (OK, ACLS)
- `C(ψ)` is the **compression** coefficient:
  - Standard: `psi * (1-psi)`
  - Weighted: `w(psi) * psi * (1-psi)`

### Non-conservative (directional form)

```
∂ψ/∂τ = n · grad(S(ψ, |grad(ψ)|))
```

where `S = ε|∇ψ| − ψ(1−ψ)` (CLS 2010).

### Algebraic source (reaction-diffusion)

```
∂ψ/∂τ = K * ψ(1−ψ)(1−2ψ) * |∇ψ|
```

where `K = 1/(4ε²)` (PM).

### Localized algebraic source (LPM)

```
∂ψ/∂τ = 4ψ(1−ψ)(1−2ψ) * [|∇ψ| − ψ(1−ψ)/ε]
```

Derived from LCLS via PM approximations (13),(15). Quartic localization.

### Potential flux (conservative first-order)

```
∂ψ/∂τ + div(g(ψ) · n̂) = 0,  g(ψ) = ψ(1−ψ)[ψ(1−ψ) − ε]
```

First-order conservative form (CFO). No normal vectors in the potential g.

## 6. Free Parameters

| Parameter | Typical range | Used by |
|-----------|--------------|---------|
| `eps` | 1e-6 to 0.1 | All methods |
| `strength` | 0.1 to 100 | All methods |
| `n_substeps` | 1 to 50 | All methods |
| `mapping_alpha` | 1.5 to 5 | CLS 2010 |
| `mapping_gamma` | 1.5 to 5 | CLS 2015 |
| `scls_alpha` | 1e-4 to 1e-2 | SCLS |
| `scls_beta` | 100 to 10000 | SCLS |

## 7. Composition Rules

A valid RHS proposal must:

1. Be expressible as one of the three templates above (conservative, non-conservative, or algebraic).
2. Specify which **normal** is used.
3. Specify the **diffusion** and **compression** components separately.
4. State whether it is **conservative** (divergence of a flux) or not.
5. List any **free parameters** with suggested defaults.
6. State **claimed invariants** (boundedness, conservation, monotonicity).
7. State **expected failure modes** (oscillation conditions, CFL limits).

## 8. Hybrid Forms (Encouraged)

New proposals may combine elements:
- PM-like source + CLS diffusion (algebraic sharpening with conservative regularization)
- Localized weight from one method + normal from another
- Adaptive ε (function of local |∇ψ|)
- Higher-order diffusion (biharmonic ∇⁴ψ for antidiffusion control)

The grammar is intentionally finite so proposals remain interpretable and
implementable as registered sharpening methods in `intsharp/sharpening.py`.
