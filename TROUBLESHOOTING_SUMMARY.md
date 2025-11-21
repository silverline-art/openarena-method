# Step Detection Troubleshooting Summary (v1.3.2)

## Problem Identified

Severe stride count inconsistency across limbs for same walking periods:
- **Before fix (v1.3.0)**: paw_RR=2, paw_RL=1, paw_FR=10, paw_FL=5 strides
- **After v1.3.2 global threshold**: paw_RR=1, paw_RL=0, paw_FR=5, paw_FL=4 strides
- **Expected**: All 4 limbs should have ~equal stride counts (quadruped walking)

## Root Cause Analysis

1. **Per-limb threshold variability** (v1.3.0 issue):
   - Each walking window computed its own adaptive threshold
   - Windows with low prominence → very low threshold → detected noise
   - Windows with high prominence → high threshold → missed actual steps
   - Solution: ✅ Implemented global threshold across all windows per limb

2. **Cross-limb signal variance** (v1.3.2 remaining issue):
   - Each limb computes its OWN global threshold
   - paw_RL: 0.033 cm/s vs paw_FR: 0.300 cm/s (9x difference!)
   - Different limbs have different movement amplitudes/clarity
   - Single prominence_multiplier can't compensate
   - **Core problem**: Need ONE threshold that works for ALL limbs

## Solutions Implemented

### v1.3.2 Features:
1. ✅ Global threshold computation (analyzes all windows together per limb)
2. ✅ Minimum threshold floors (0.3 cm/s bottom, 0.5 cm side view)
3. ✅ Auto-calibration loop (tests prominence_multiplier [0.2-0.8])
4. ✅ View-specific detection methods (bottom vs vertical)
5. ✅ Helper methods for consistent threshold application

### Files Modified:
- `src/exmo_gait/analysis/step_detector.py`: Added global threshold computation
- `src/exmo_gait/analysis/parameter_calibrator.py`: Auto-calibration class (NEW)
- `src/exmo_gait/pipeline/stages.py`: Integrated auto-calibration

## Next Steps Required

### Critical Fix Needed:
Implement **cross-limb unified threshold**:
1. Pool ALL limbs' data together
2. Compute ONE prominence threshold for all paws
3. Apply same threshold to all 4 limbs
4. This ensures consistent detection across limbs

### Alternative Approach:
**Consensus-based detection**:
1. For each candidate foot strike frame
2. Check if ≥3 limbs detected strike within ±2 frames
3. If yes, mark as valid stride event for all limbs
4. This enforces quadruped coordination constraint

### Validation Metrics:
- Coefficient of variation (CV) across limbs should be <20%
- All limbs should have within ±2 strides of each other
- Cadence should be 100-200 steps/min for mice

## Test Data Analysis

Sample: `Top_irradiated_open_control_5_main_20250716_102813.csv`
- 8 walking windows detected
- Total duration: 60.6 seconds
- Expected strides per limb: ~10-15 (based on typical mouse cadence)
- Actual results: Highly variable (0-5 strides per limb)

**Conclusion**: Current approach improved consistency within each limb but didn't solve cross-limb variance. Need unified threshold or consensus mechanism.
