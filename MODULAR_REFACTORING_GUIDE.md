# Modular Refactoring Guide

## Overview
This guide helps break down large files (>2000 lines) into smaller, modular components while preserving all existing logic.

## Current Large Files
1. `src/ecg/twelve_lead_test.py` - 8276 lines → Split into 4-5 modules
2. `src/dashboard/dashboard.py` - 3891 lines → Split into 2-3 modules  
3. `src/ecg/ecg_report_generator.py` - 3815 lines → Split into 2 modules
4. `src/ecg/hrv_ecg_report_generator.py` - 5220 lines → Split into 3 modules
5. `src/ecg/hyperkalemia_ecg_report_generator.py` - 4689 lines → Split into 3 modules
6. `src/ecg/recording.py` - 2611 lines → Split into 2 modules
7. `src/ecg/expanded_lead_view.py` - 2294 lines → Split into 2 modules

## New Folder Structure Created

```
src/
├── ecg/
│   ├── ui/              # UI components
│   ├── serial/          # Serial communication
│   ├── metrics/         # Metrics calculation
│   ├── plotting/        # Plotting utilities
│   └── utils/           # Utilities (constants, helpers)
├── dashboard/
│   ├── ui/              # UI components
│   ├── widgets/         # Reusable widgets
│   └── metrics/         # Metrics handling
└── reports/
    ├── generators/      # Report generators
    └── templates/       # Report templates
```

## Step-by-Step Refactoring Process

### Phase 1: Extract Utilities (DONE)
- ✅ Created `ecg/utils/constants.py` - Constants and configuration
- ✅ Created `ecg/utils/helpers.py` - Helper functions

### Phase 2: Extract Serial Communication
**Target:** `src/ecg/serial/serial_reader.py` and `packet_parser.py`

**Extract from `twelve_lead_test.py`:**
- Lines 234-298: Packet parsing functions (`parse_packet`, `decode_lead`, `hex_string_to_bytes`)
- Lines 300-531: `SerialStreamReader` class
- Lines 536-695: `SerialECGReader` class

**Steps:**
1. Create `src/ecg/serial/packet_parser.py` with packet parsing functions
2. Create `src/ecg/serial/serial_reader.py` with SerialStreamReader and SerialECGReader
3. Update imports in `twelve_lead_test.py` to use new modules

### Phase 3: Extract Metrics Calculation
**Target:** `src/ecg/metrics/` modules

**Extract from `twelve_lead_test.py`:**
- `calculate_ecg_metrics()` method (~lines 2000-2500)
- Axis calculation methods
- Heart rate calculation
- Interval calculations (PR, QRS, QT)

**Create:**
- `ecg/metrics/heart_rate.py` - HR calculation
- `ecg/metrics/intervals.py` - PR, QRS, QT calculations  
- `ecg/metrics/axis.py` - Axis calculations
- `ecg/metrics/metrics_manager.py` - Main metrics orchestration

### Phase 4: Extract Plotting Logic
**Target:** `src/ecg/plotting/` modules

**Extract from `twelve_lead_test.py`:**
- Plot initialization methods
- Plot update methods
- PyQtGraph configuration

**Create:**
- `ecg/plotting/pyqtgraph_plots.py` - PyQtGraph plot widgets
- `ecg/plotting/plot_utils.py` - Plotting utilities

### Phase 5: Split Main ECGTestPage Class
**Target:** `src/ecg/ui/ecg_test_page.py` (main class)

**Keep in main file:**
- `ECGTestPage.__init__()` - Initialization
- UI setup methods
- Event handlers

**Move to separate modules:**
- Display update logic → `ecg/ui/ecg_display.py`
- Metrics display → `ecg/ui/ecg_metrics_display.py`

### Phase 6: Split Dashboard
**Target:** `src/dashboard/ui/` modules

**Split `dashboard.py`:**
- Main Dashboard class → `dashboard/ui/dashboard_main.py`
- Widget creation → `dashboard/ui/dashboard_widgets.py`
- Metrics handling → `dashboard/metrics/metrics_manager.py`

### Phase 7: Split Report Generators
**Target:** `src/reports/generators/` modules

Each report generator can be split into:
- Base generator class → `reports/generators/base_generator.py`
- Report-specific logic → Separate files per report type

## Migration Strategy

### Backward Compatibility Approach
1. Create new modules with extracted code
2. Update original files to import from new modules
3. Keep original files as thin wrappers initially
4. Gradually remove wrapper code once all imports are updated

### Example Migration Pattern

**Before:**
```python
# twelve_lead_test.py (8276 lines)
class SerialStreamReader:
    # ... 200 lines of code ...
```

**After:**
```python
# ecg/serial/serial_reader.py (new file)
class SerialStreamReader:
    # ... 200 lines of code ...

# twelve_lead_test.py (updated)
from ecg.serial import SerialStreamReader
# ... rest of code ...
```

## Testing Checklist
- [ ] All imports resolve correctly
- [ ] No circular import issues
- [ ] All functionality preserved
- [ ] UI still works correctly
- [ ] Serial communication works
- [ ] Metrics calculation works
- [ ] Report generation works

## Tools and Scripts
- `refactor_modules.py` - Analysis script (needs fixes)
- Use IDE refactoring tools for safe code movement
- Test after each module extraction

## Notes
- Maintain all existing logic - no changes to algorithms
- Use relative imports within packages
- Keep class interfaces unchanged
- Preserve all method signatures
