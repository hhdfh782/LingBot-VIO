# Checkpoints: post-acceptance release

Model weights are intentionally not distributed during peer review. The complete authorized checkpoint set will be released after paper acceptance.

For every released file, publish:

| File | Purpose | SHA-256 | Download |
|---|---|---|---|
| `lingbot_vio.ckpt` | Main Fusion checkpoint | Post-acceptance | Post-acceptance |
| `lingbot_map.pt` | Visual baseline/backbone | Post-acceptance | Upstream or authorized mirror |
| `airimu.ckpt` | IMU encoder initialization | Post-acceptance | Upstream or authorized mirror |

Verify a download with:

```bash
sha256sum checkpoints/lingbot_vio.ckpt
```

Windows PowerShell:

```powershell
Get-FileHash checkpoints/lingbot_vio.ckpt -Algorithm SHA256
```
