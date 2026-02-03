# intsharp

**Documentation:** [https://intsharp.marziale.tech](https://intsharp.marziale.tech)

Interface-sharpening and compressible Euler simulation. Supports:
- **Advection mode**: Scalar advection with Chiu-Lin or Parameswaran-Mandal sharpening
- **Euler mode**: 1D compressible flow (single-phase or two-phase 5-equation model)

## Quick start

```bash
pip install -r requirements.txt
python run.py unit_tests/sod_shock_tube_1d.yaml
pytest tests/ -v
```
