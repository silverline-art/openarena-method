# EXMO Gait Analysis Pipeline - Documentation Index
**Last Updated:** 2025-11-21
**Version:** v1.2.0 Calibrated

---

## Quick Navigation

### For Immediate Action
1. **[AUDIT_EXECUTIVE_SUMMARY.md](AUDIT_EXECUTIVE_SUMMARY.md)** - Start here (5 min read)
2. **[QUICK_REFERENCE_CORRECTIONS.md](QUICK_REFERENCE_CORRECTIONS.md)** - Parameter changes (10 min read)
3. **Run validation:** `python docs/validation_calculations.py`

### For Deep Understanding
4. **[METRIC_CORRECTNESS_AUDIT_REPORT.md](METRIC_CORRECTNESS_AUDIT_REPORT.md)** - Complete analysis (30 min read)
5. **[ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md)** - System design (20 min read)

---

## Document Overview

### Critical Documents (READ FIRST)

#### 1. AUDIT_EXECUTIVE_SUMMARY.md
**Size:** 9.5 KB | **Read Time:** 5 minutes

**Purpose:** High-level overview of metric correctness issues and fixes

**Contents:**
- 6 critical issues identified
- Impact by metric (table)
- Before/after parameter comparison
- Deployment checklist
- Risk assessment
- Timeline

**Who should read:** Everyone (PIs, developers, analysts)

---

#### 2. QUICK_REFERENCE_CORRECTIONS.md
**Size:** 9.2 KB | **Read Time:** 10 minutes

**Purpose:** Practical guide for using v1.2.0 corrections

**Contents:**
- TL;DR parameter changes
- Code changes (before/after)
- Expected metric changes
- When to use which version
- Troubleshooting guide
- Quick commands

**Who should read:** Developers, analysts running analyses

---

#### 3. METRIC_CORRECTNESS_AUDIT_REPORT.md
**Size:** 36 KB | **Read Time:** 30 minutes

**Purpose:** Comprehensive root cause analysis with mathematical validation

**Contents:**
- Detailed analysis of 6 systematic biases:
  1. Scaling factor (25% underestimation)
  2. Smoothing bias (15-30% dampening)
  3. Walking threshold (56% false negatives)
  4. Stride rejection (44% valid stride loss)
  5. ROM calculation (hip was broken)
  6. COM speed (15% underestimation)
- Mathematical validation for each issue
- Sample calculations
- Expected improvements
- Recommendations

**Who should read:** Technical leads, method validation, publication authors

**Sections:**
- Executive Summary
- 6 Root Cause Analyses (one per issue)
- Integrated Impact Analysis
- Validation Protocol
- Recommendations
- Appendices

---

### System Documentation

#### 4. SYSTEM_OVERVIEW.md
**Size:** 34 KB | **Read Time:** 25 minutes

**Purpose:** Complete pipeline overview and usage guide

**Contents:**
- Installation guide
- Architecture overview
- Data format specifications
- Processing pipeline flow
- Batch processing guide
- Output formats
- Troubleshooting

**Who should read:** New users, system administrators

---

#### 5. ARCHITECTURE_REVIEW.md
**Size:** 58 KB | **Read Time:** 40 minutes

**Purpose:** Deep dive into system design and implementation

**Contents:**
- Component architecture
- Data flow analysis
- Module responsibilities
- Interface contracts
- Design patterns
- Extensibility points
- Performance characteristics

**Who should read:** Developers, system architects

---

#### 6. ARCHITECTURE_DIAGRAM.txt
**Size:** 16 KB | **Read Time:** 10 minutes

**Purpose:** Visual ASCII diagram of system architecture

**Contents:**
- Layer architecture diagram
- Data flow visualization
- Component relationships
- Processing pipeline stages

**Who should read:** Visual learners, presentations

---

#### 7. API_REFERENCE.md
**Size:** 23 KB | **Read Time:** Reference

**Purpose:** Complete API documentation

**Contents:**
- Class documentation
- Function signatures
- Parameter descriptions
- Return types
- Usage examples

**Who should read:** Developers, API users

---

#### 8. README.md
**Size:** 8.0 KB | **Read Time:** 10 minutes

**Purpose:** Project introduction and quick start

**Contents:**
- Project overview
- Key features
- Installation
- Quick start examples
- Documentation links

**Who should read:** First-time users

---

### Validation Tools

#### 9. validation_calculations.py
**Size:** 32 KB | **Type:** Executable Python script

**Purpose:** Demonstrate and validate metric corrections

**Features:**
- Scaling factor validation (v1.1.0 vs v1.2.0)
- Smoothing impact analysis
- Walking threshold comparison
- Stride detection validation
- ROM calculation verification
- COM speed 2D vs 3D comparison
- Comprehensive comparison report

**Usage:**
```bash
python docs/validation_calculations.py
```

**Output:**
- Mathematical demonstrations
- Sample calculations
- Before/after comparisons
- Validation report

**Who should run:** Developers validating changes, analysts verifying results

---

## Document Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                    AUDIT DOCUMENTS                          │
│                                                             │
│  AUDIT_EXECUTIVE_SUMMARY.md                                │
│           ↓                                                 │
│  QUICK_REFERENCE_CORRECTIONS.md                            │
│           ↓                                                 │
│  METRIC_CORRECTNESS_AUDIT_REPORT.md                        │
│           ↓                                                 │
│  validation_calculations.py                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  SYSTEM DOCUMENTS                           │
│                                                             │
│  README.md                                                  │
│           ↓                                                 │
│  SYSTEM_OVERVIEW.md                                         │
│           ↓                                                 │
│  ARCHITECTURE_REVIEW.md                                     │
│           ↓                                                 │
│  ARCHITECTURE_DIAGRAM.txt                                   │
│           ↓                                                 │
│  API_REFERENCE.md                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Reading Paths by Role

### Principal Investigator / Lab Manager
**Goal:** Understand impact and decide on deployment

1. **AUDIT_EXECUTIVE_SUMMARY.md** (5 min)
   - Critical issues overview
   - Impact assessment
   - Deployment timeline

2. **QUICK_REFERENCE_CORRECTIONS.md** (10 min)
   - Expected metric changes
   - When to use which version

**Total Time:** 15 minutes

**Decision Point:** Approve v1.2.0 deployment after validation testing

---

### Data Analyst / Computational Biologist
**Goal:** Process data correctly and interpret results

1. **QUICK_REFERENCE_CORRECTIONS.md** (10 min)
   - Parameter changes
   - Expected metric changes
   - Troubleshooting

2. **SYSTEM_OVERVIEW.md** (25 min)
   - Pipeline usage
   - Batch processing
   - Output interpretation

3. **Run:** `python docs/validation_calculations.py` (5 min)
   - Validate installation
   - Understand corrections

**Total Time:** 40 minutes

**Action Items:**
- Use `config_v1.2_calibrated.yaml` for new analyses
- Reprocess critical historical datasets
- Document metric changes in publications

---

### Software Developer / Method Developer
**Goal:** Understand implementation and contribute code

1. **AUDIT_EXECUTIVE_SUMMARY.md** (5 min)
   - Issues overview

2. **METRIC_CORRECTNESS_AUDIT_REPORT.md** (30 min)
   - Root cause analysis
   - Mathematical validation
   - Code changes

3. **ARCHITECTURE_REVIEW.md** (40 min)
   - System design
   - Module structure
   - Extension points

4. **API_REFERENCE.md** (reference)
   - Function signatures
   - Parameter types

5. **Study:** `validation_calculations.py` (30 min)
   - Implementation examples
   - Test patterns

**Total Time:** 2 hours

**Action Items:**
- Implement additional validations
- Write unit tests for corrections
- Contribute improvements

---

### Publication Author / Reviewer
**Goal:** Validate methods and interpret results

1. **METRIC_CORRECTNESS_AUDIT_REPORT.md** (30 min)
   - Complete root cause analysis
   - Mathematical validation
   - Expected corrections

2. **QUICK_REFERENCE_CORRECTIONS.md** (10 min)
   - Parameter summary
   - Version comparison

3. **SYSTEM_OVERVIEW.md** - Methods section (15 min)
   - Algorithm descriptions
   - Parameter justifications

**Total Time:** 55 minutes

**Action Items:**
- Update methods sections
- Reanalyze critical data with v1.2.0
- Compare v1.1.0 vs v1.2.0 results
- Document corrections in publications

---

## Configuration Files

### Production Use (RECOMMENDED)
```
/config_v1.2_calibrated.yaml
```
- Corrected scaling (10cm reference)
- Reduced smoothing (7-frame position, EMA velocity)
- Relaxed thresholds (MAD 0.8, hybrid method)
- Expanded stride detection (0.06s minimum)
- 3D COM calculation
- Minimal ROM smoothing

**Use for:** All new analyses, reprocessing critical data

---

### Legacy Compatibility
```
/config.yaml
```
- Original parameters (8cm scaling, 11-frame smoothing)
- Strict thresholds (MAD 2.0)
- 0.1s minimum stride duration
- 2D COM only

**Use for:** Side-by-side comparison with historical data only

---

### Adaptive Configuration
```
/config_adaptive.yaml
```
- Intermediate version (v1.1.5)
- Some corrections applied, not all

**Use for:** Transitional testing (not recommended for production)

---

## Quick Start Workflow

### 1. First Time Setup (15 min)
```bash
# Read executive summary
cat docs/AUDIT_EXECUTIVE_SUMMARY.md

# Read quick reference
cat docs/QUICK_REFERENCE_CORRECTIONS.md

# Run validation script
python docs/validation_calculations.py
```

### 2. Process Sample Data (10 min)
```bash
# Single sample with v1.2.0
python batch_process.py \
  --config config_v1.2_calibrated.yaml \
  --sample control_5

# Check output
ls -la Output/control/control_5/
```

### 3. Compare Versions (20 min)
```bash
# Run both versions side-by-side
python batch_process.py \
  --config config_v1.2_calibrated.yaml \
  --compare-baseline config.yaml \
  --sample control_5 \
  --output comparison/

# Review differences
cat comparison/control_5/metric_comparison.txt
```

### 4. Validation Testing (2-3 days)
```bash
# Batch process control group with both versions
python batch_process.py \
  --config config_v1.2_calibrated.yaml \
  --compare-baseline config.yaml \
  --group control \
  --parallel 4

# Manual annotation validation
# (See METRIC_CORRECTNESS_AUDIT_REPORT.md Section 8.3)
```

### 5. Production Deployment (1 day)
```bash
# Archive v1.1.0 results
mv Output/ Output_v1.1.0_archive/

# Deploy v1.2.0
cp config_v1.2_calibrated.yaml config_production.yaml

# Batch process all groups
python batch_process.py \
  --config config_production.yaml \
  --batch
```

---

## Support & Troubleshooting

### Common Issues

**Issue:** Metrics look too high after v1.2.0
**Solution:** Expected. v1.1.0 systematically underestimated. See QUICK_REFERENCE_CORRECTIONS.md

**Issue:** Too many strides detected
**Solution:** Adjust `min_stride_duration` or `prominence_multiplier`. See troubleshooting section.

**Issue:** Walking detection too aggressive
**Solution:** Adjust `walking_mad_threshold` or `adaptive_percentile`. See troubleshooting section.

**Issue:** Hip ROM still >100°
**Solution:** Check keypoint mapping. Should use hip_center → hip → knee, not hip → paw.

---

## Version History

### v1.2.0 (2025-11-21) - CURRENT
- **MAJOR:** Corrected scaling factor (8cm → 10cm)
- **MAJOR:** Reduced smoothing (11 → 7/5/3 frames)
- **MAJOR:** Relaxed walking threshold (MAD 2.0 → 0.8)
- **MAJOR:** Expanded stride detection (0.1s → 0.06s)
- **FIX:** Hip ROM keypoints corrected
- **NEW:** 3D COM calculation
- **NEW:** Hybrid threshold method
- **NEW:** Micro-step labeling
- **Documentation:** Complete audit report

### v1.1.5 (2025-11-21) - TRANSITIONAL
- Partial corrections applied
- Not recommended for production

### v1.1.0 (2025-11-20) - LEGACY
- Original implementation
- Systematic underestimation issues
- Use only for historical comparison

---

## Citation

If using this software, please cite:

```
EXMO Gait Analysis Pipeline v1.2.0
[Your Institution]
DOI: [To be assigned]
```

For v1.2.0 metric corrections, also cite:
```
Metric Correctness Audit Report (2025)
Performance Engineering Analysis
Available: docs/METRIC_CORRECTNESS_AUDIT_REPORT.md
```

---

## Contact & Contributions

**Documentation Issues:** Open GitHub issue or contact [maintainer]

**Method Questions:** Review audit report first, then contact [lead scientist]

**Code Contributions:** Follow development guidelines in ARCHITECTURE_REVIEW.md

**Bug Reports:** Include:
- Configuration file used
- Sample data characteristics
- Error messages
- Expected vs actual behavior

---

**Document Index Version:** 1.0
**Last Updated:** 2025-11-21
**Maintained By:** EXMO Development Team

---
