# Unit tests / validation cases

Hyperbolic tangent (tanh-hat) validation: **100 pt mesh**, **one revolution** in a **periodic** domain, **10-frame GIF** output.

| Config | Description | Output |
|--------|-------------|--------|
| `tanh_one_rev_no_sharpening.yaml` | Advection only, no sharpening | `results_no_sharpening/alpha.gif` |
| `tanh_one_rev_cl_sharpening.yaml` | Advection + CL (Chiu–Lin) sharpening | `results_cl_sharpening/alpha.gif` |

- **One revolution:** domain length L = 1 (x ∈ [-0.5, 0.5]), velocity u = 0.5 ⇒ T = L/|u| = 2.0 s ⇒ 200 steps × dt 0.01.
- **10 frames:** GIF monitor `every_n_steps: 20` ⇒ 200/20 = 10 frames.

Run from repo root:

```bash
python run.py unit_tests/tanh_one_rev_no_sharpening.yaml
python run.py unit_tests/tanh_one_rev_cl_sharpening.yaml
```
