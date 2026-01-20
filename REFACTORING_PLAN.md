# Modular Refactoring Plan

## Target Structure

```
src/
├── ecg/
│   ├── __init__.py
│   ├── ui/                    # UI components
│   │   ├── __init__.py
│   │   ├── ecg_test_page.py  # Main ECGTestPage class (~2000 lines)
│   │   ├── ecg_plotting.py   # Plotting logic (~2000 lines)
│   │   ├── ecg_metrics.py    # Metrics calculation (~2000 lines)
│   │   └── ecg_display.py    # Display updates (~2000 lines)
│   ├── serial/               # Serial communication
│   │   ├── __init__.py
│   │   ├── serial_reader.py # SerialECGReader class
│   │   └── packet_parser.py  # Packet parsing utilities
│   ├── metrics/              # Metrics calculation
│   │   ├── __init__.py
│   │   ├── heart_rate.py     # HR calculation
│   │   ├── intervals.py      # PR, QRS, QT calculations
│   │   └── axis.py           # Axis calculations
│   ├── plotting/             # Plotting utilities
│   │   ├── __init__.py
│   │   ├── pyqtgraph_plots.py
│   │   └── plot_utils.py
│   └── utils/                # Utilities (already created)
│       ├── __init__.py
│       ├── constants.py
│       └── helpers.py
├── dashboard/
│   ├── ui/                   # UI components
│   │   ├── __init__.py
│   │   ├── dashboard_main.py # Main Dashboard class (~2000 lines)
│   │   └── dashboard_widgets.py # Widget components (~2000 lines)
│   ├── widgets/              # Reusable widgets
│   │   ├── __init__.py
│   │   ├── metrics_display.py
│   │   └── heart_widget.py
│   └── metrics/              # Metrics handling
│       ├── __init__.py
│       └── metrics_manager.py
└── reports/
    ├── generators/           # Report generators
    │   ├── __init__.py
    │   ├── base_generator.py
    │   ├── standard_report.py
    │   ├── hrv_report.py
    │   └── hyperkalemia_report.py
    └── templates/            # Report templates
        └── __init__.py
```

## File Breakdown Strategy

### twelve_lead_test.py (8276 lines) → Split into:

1. **ecg/ui/ecg_test_page.py** (~2000 lines)
   - ECGTestPage class __init__
   - UI setup methods
   - Event handlers

2. **ecg/ui/ecg_plotting.py** (~2000 lines)
   - Plot initialization
   - Plot update methods
   - PyQtGraph configuration

3. **ecg/ui/ecg_metrics.py** (~2000 lines)
   - calculate_ecg_metrics()
   - Metrics calculation methods
   - Axis calculations

4. **ecg/serial/serial_reader.py** (~1500 lines)
   - SerialECGReader class
   - SerialStreamReader class
   - Serial communication logic

5. **ecg/ui/ecg_display.py** (~1500 lines)
   - Display update methods
   - UI refresh logic
   - Metric display updates

### dashboard.py (3891 lines) → Split into:

1. **dashboard/ui/dashboard_main.py** (~2000 lines)
   - Dashboard class __init__
   - Main UI setup
   - Core functionality

2. **dashboard/ui/dashboard_widgets.py** (~2000 lines)
   - Widget creation methods
   - Layout management
   - UI components

## Implementation Notes

- All imports must be updated
- All class references must be preserved
- No logic changes - only structural reorganization
- Use relative imports within packages
- Maintain backward compatibility where possible
