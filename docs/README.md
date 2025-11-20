# EXMO Gait Analysis Pipeline - Documentation Index

## Quick Links

- **[Quick Start Guide](../QUICK_START.md)** - Get up and running in 5 minutes
- **[System Overview](SYSTEM_OVERVIEW.md)** - Architecture and core concepts
- **[API Reference](API_REFERENCE.md)** - Developer documentation
- **[Troubleshooting](../TROUBLESHOOTING.md)** - Common issues and solutions

---

## Documentation Structure

### User Guides

| Document | Description | Audience |
|----------|-------------|----------|
| [QUICK_START.md](../QUICK_START.md) | Installation and first analysis | New users |
| [BATCH_PROCESSING.md](../BATCH_PROCESSING.md) | Batch processing guide | All users |
| [VISUALIZATION_UPGRADE.md](../VISUALIZATION_UPGRADE.md) | Publication-grade plots | Researchers |
| [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) | Issue resolution | All users |

### Technical Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) | Complete system architecture | Developers |
| [API_REFERENCE.md](API_REFERENCE.md) | API documentation | Developers |
| [VISUALIZATION_PRD_IMPLEMENTATION.md](../VISUALIZATION_PRD_IMPLEMENTATION.md) | PRD compliance report | Project managers |

### Specific Fixes and Enhancements

| Document | Description | Version |
|----------|-------------|---------|
| [ADAPTIVE_THRESHOLD_FIX.md](../ADAPTIVE_THRESHOLD_FIX.md) | Empty plots fix | v1.1.0 |
| [BATCH_PROCESS_UPGRADE.md](../BATCH_PROCESS_UPGRADE.md) | Batch processing v1.1 | v1.1.0 |
| [FIX_CONTROL_SAMPLES.md](../FIX_CONTROL_SAMPLES.md) | Missing samples fix | v1.1.0 |

---

## Quick Reference

### Common Tasks

**Process Single Sample**:
```bash
python batch_process.py --config config_adaptive.yaml --sample control_5
```

**Process Group**:
```bash
python batch_process.py --config config_adaptive.yaml --group control --parallel 4
```

**Process All Samples**:
```bash
python batch_process.py --config config_adaptive.yaml --batch --parallel 8
```

**Dry-Run Preview**:
```bash
python batch_process.py --config config_adaptive.yaml --batch --dry-run
```

**Diagnose Thresholds**:
```bash
python diagnose_thresholds.py --sample control_5 --config config.yaml
```

### Configuration Quick Reference

**Standard Mode** (config.yaml):
- Use for high-activity datasets (treadmill, forced locomotion)
- walking_mad_threshold: 2.0
- plot_dpi: 300

**Adaptive Mode** (config_adaptive.yaml):
- Use for low-activity datasets (open field, exploratory)
- walking_mad_threshold: 1.2 (40% more sensitive)
- plot_dpi: 600 (publication quality)
- adaptive_thresholding: true

### Output Structure

```
Output/
├── {group_name}/
│   ├── {sample_id}/
│   │   ├── Gait_Analysis_{sample_id}_{timestamp}.xlsx
│   │   ├── plot_coordination.png
│   │   ├── plot_speed_spatial.png
│   │   ├── plot_phase_timing.png
│   │   ├── plot_range_of_motion.png
│   │   └── intermediates/
│   │       ├── gait_metrics.npz
│   │       ├── rom_metrics.npz
│   │       └── phase_windows.npz
└── Batch_Summary_{timestamp}.xlsx
```

### Key Metrics Explained

**Temporal Metrics**:
- **Cadence**: Steps per minute (normal: 180-240 for mice)
- **Stride Time**: Time between successive foot strikes
- **Duty Cycle**: Stance time / (stance + swing) × 100 (normal: 50-70%)

**Spatial Metrics**:
- **Stride Length**: Distance traveled in one stride (cm)
- **Step Width**: Lateral distance between paws

**Coordination Metrics**:
- **Regularity Index**: Measure of diagonal pair coordination (0-1, higher = better)
- **Phase Dispersion**: Variability in limb coordination

**ROM Metrics**:
- **Joint Angles**: Hip, knee, elbow angles in degrees
- **Angular Velocity**: Rate of joint angle change (deg/s)
- **CoM Sway**: Body center of mass movement (ML/AP)

---

## For Different User Types

### New Users
1. Read [QUICK_START.md](../QUICK_START.md)
2. Process one sample to verify installation
3. Review [VISUALIZATION_UPGRADE.md](../VISUALIZATION_UPGRADE.md) for plot interpretation

### Researchers
1. Review [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) for methodology
2. Configure thresholds using [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
3. Use [BATCH_PROCESSING.md](../BATCH_PROCESSING.md) for large datasets
4. Generate publication plots with config_adaptive.yaml

### Developers
1. Study [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) for architecture
2. Reference [API_REFERENCE.md](API_REFERENCE.md) for integration
3. Review source code in `src/exmo_gait/`

### Lab Managers
1. [QUICK_START.md](../QUICK_START.md) for setup
2. [BATCH_PROCESSING.md](../BATCH_PROCESSING.md) for workflow
3. [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) for common issues

---

## Troubleshooting Quick Links

**Empty Plots**: See [ADAPTIVE_THRESHOLD_FIX.md](../ADAPTIVE_THRESHOLD_FIX.md)

**Missing Samples**: See [FIX_CONTROL_SAMPLES.md](../FIX_CONTROL_SAMPLES.md)

**Low Quality Data**: Check preprocessing parameters in [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)

**Slow Processing**: Use parallel mode, see [BATCH_PROCESS_UPGRADE.md](../BATCH_PROCESS_UPGRADE.md)

---

## Version-Specific Documentation

### v1.1.0 (Current)
- ✅ Adaptive thresholding system
- ✅ Publication-grade visualization (600 DPI)
- ✅ Enhanced batch processing
- ✅ Dry-run mode
- ✅ Processing time metrics

**New in v1.1.0**:
- [ADAPTIVE_THRESHOLD_FIX.md](../ADAPTIVE_THRESHOLD_FIX.md)
- [VISUALIZATION_UPGRADE.md](../VISUALIZATION_UPGRADE.md)
- [BATCH_PROCESS_UPGRADE.md](../BATCH_PROCESS_UPGRADE.md)
- [FIX_CONTROL_SAMPLES.md](../FIX_CONTROL_SAMPLES.md)

### v1.0.0 (Initial Release)
- Basic gait analysis
- Multi-view integration
- Excel export
- Standard visualization

---

## FAQ

**Q: Which config should I use?**
A: Use `config_adaptive.yaml` for low-activity datasets (open field) or `config.yaml` for high-activity datasets (treadmill).

**Q: How do I generate publication-quality plots?**
A: Set `use_enhanced_plots: true` and `plot_dpi: 600` in your config file.

**Q: What if I get empty plots?**
A: Use `config_adaptive.yaml` or run `diagnose_thresholds.py` to find optimal thresholds.

**Q: How many CPU cores should I use for parallel processing?**
A: Use `--parallel N` where N = number of physical cores - 1 (e.g., `--parallel 7` for 8-core CPU).

**Q: Can I process samples with different file naming?**
A: Yes! Update `file_patterns` in config.yaml. Current patterns support `_main`, `_test`, and other suffixes.

**Q: How do I interpret the regularity index?**
A: Values near 1.0 indicate good diagonal limb coordination, values below 0.8 suggest coordination deficits.

---

## Getting Help

**Documentation Issues**: Open an issue on the project repository

**Processing Errors**: Check [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) first

**Feature Requests**: See [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) roadmap section

**Bug Reports**: Include config file, sample data structure, and error messages

---

## Contributing to Documentation

When updating documentation:

1. **Keep it concise**: Users want answers, not essays
2. **Add examples**: Code examples are worth 1000 words
3. **Update index**: Add new docs to this README.md
4. **Version tag**: Note which version the doc applies to
5. **Cross-reference**: Link related documents

---

## Document Status

| Document | Last Updated | Status | Version |
|----------|--------------|--------|---------|
| QUICK_START.md | 2025-11-21 | ✅ Current | v1.1.0 |
| SYSTEM_OVERVIEW.md | 2025-11-21 | ✅ Current | v1.1.0 |
| API_REFERENCE.md | 2025-11-21 | ✅ Current | v1.1.0 |
| BATCH_PROCESSING.md | 2025-11-21 | ✅ Current | v1.1.0 |
| TROUBLESHOOTING.md | 2025-11-21 | ✅ Current | v1.1.0 |
| VISUALIZATION_UPGRADE.md | 2025-11-21 | ✅ Current | v1.1.0 |
| ADAPTIVE_THRESHOLD_FIX.md | 2025-11-21 | ✅ Current | v1.1.0 |
| BATCH_PROCESS_UPGRADE.md | 2025-11-21 | ✅ Current | v1.1.0 |
| FIX_CONTROL_SAMPLES.md | 2025-11-21 | ✅ Current | v1.1.0 |

---

**Documentation Version**: 1.1.0
**Last Updated**: 2025-11-21
**Status**: ✅ Complete and Current
