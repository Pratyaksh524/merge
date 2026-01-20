from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QSizePolicy,
    QApplication,
    QFileDialog,
)
from PyQt5.QtCore import Qt
import os
import json
import datetime
import sys
import shutil


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HISTORY_FILE = os.path.join(BASE_DIR, "ecg_history.json")
ECG_DATA_FILE = os.path.join(BASE_DIR, "ecg_data.txt")
REPORTS_INDEX_FILE = os.path.join(BASE_DIR, "reports", "index.json")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


class HistoryWindow(QDialog):
    """ECG reports history: shows one row per generated report with basic patient details."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ECG Report History")
        
        # Make window responsive to screen size
        screen = QApplication.desktop().screenGeometry()
        window_width = int(screen.width() * 0.8)
        window_height = int(screen.height() * 0.7)
        self.resize(window_width, window_height)
        
        # Set minimum size for usability
        self.setMinimumSize(800, 400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels(
            [
                "Date",
                "Time",
                "Org.",
                "Doctor",
                "Patient Name",
                "Age",
                "Gender",
                "Height (cm)",
                "Weight (kg)",
                "Report Type",
            ]
        )
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(self.table.SelectRows)
        self.table.setEditTriggers(self.table.NoEditTriggers)
        
        # Make table expand to fill available space
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Set column stretch factors for proportional expansion
        # Date and Time: smaller, Patient Name and Doctor: larger
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(self.table.horizontalHeader().Stretch)
        
        layout.addWidget(self.table, 1)

        # Connect double-click signal to open report
        self.table.cellDoubleClicked.connect(self.on_row_double_clicked)

        # Buttons row
        btn_row = QHBoxLayout()
        self.open_btn = QPushButton("Open Report")
        self.open_btn.setStyleSheet(
            "background: #ff6600; color: white; border-radius: 10px; padding: 6px 18px;"
        )
        self.open_btn.clicked.connect(self.open_selected_report)
        btn_row.addWidget(self.open_btn)

        self.export_all_btn = QPushButton("Export All")
        self.export_all_btn.setStyleSheet(
            "background: #28a745; color: white; border-radius: 10px; padding: 6px 18px;"
        )
        self.export_all_btn.clicked.connect(self.export_all_reports)
        btn_row.addWidget(self.export_all_btn)

        btn_row.addStretch(1)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_row.addWidget(self.close_btn)

        layout.addLayout(btn_row)

        self.load_history()
        
        # Connect resize event to update column widths
        self.table.horizontalHeader().sectionResizeMode(self.table.horizontalHeader().Stretch)

    def load_history(self):
        """Load history entries (preferring ecg_history.json) into the table."""
        self.table.setRowCount(0)

        history_entries = []

        # Preferred source: rich per-report history
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history_entries = json.load(f)
                if not isinstance(history_entries, list):
                    history_entries = []
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load history from ecg_history.json: {e}")
                history_entries = []

        # Fallback: basic patient list (older flow without report_type)
        if not history_entries:
            patients_file = os.path.join(BASE_DIR, "all_patients.json")
            if not os.path.exists(patients_file):
                return
            try:
                with open(patients_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                patients = data.get("patients", []) if isinstance(data, dict) else []
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load history from all_patients.json: {e}")
                return

            for p in patients:
                patient_name = p.get("patient_name") or (
                    (p.get("first_name", "") + " " + p.get("last_name", "")).strip()
                )
                org = p.get("Org.", "")
                doctor = p.get("doctor", "")
                age = str(p.get("age", ""))
                gender = p.get("gender", "")
                height = str(p.get("height", "")) if p.get("height", "") != "" else ""
                weight = str(p.get("weight", "")) if p.get("weight", "") != "" else ""

                date_time = p.get("date_time", "")
                date_str, time_str = "", ""
                if date_time and " " in date_time:
                    date_str, time_str = date_time.split(" ", 1)
                elif date_time:
                    date_str = date_time

                entry = {
                    "date": date_str,
                    "time": time_str,
                    "report_type": "ECG",
                    "Org.": org,
                    "doctor": doctor,
                    "patient_name": patient_name,
                    "age": age,
                    "gender": gender,
                    "height": height,
                    "weight": weight,
                    "report_file": "",
                }
                history_entries.append(entry)

        # Normalize and populate table
        for entry in history_entries:
            report_file = entry.get("report_file", "") or ""
            report_type = entry.get("report_type", "")
            if not report_type:
                file_lower = report_file.lower()
                if "hyper" in file_lower:
                    report_type = "Hyperkalemia"
                elif "hrv" in file_lower:
                    report_type = "HRV"
                elif "ecg" in file_lower:
                    report_type = "ECG"
                else:
                    report_type = "ECG"
            entry["report_type"] = report_type
            self.add_row(entry)

    def _get_report_datetime(self, patient_name, reports_index):
        """Get the actual date/time when report was generated for a patient."""
        # First, try to find in reports index
        if patient_name and patient_name in reports_index:
            patient_reports = reports_index[patient_name]
            if patient_reports:
                # Use the most recent report (first in list, as index.json is usually sorted by newest first)
                most_recent = patient_reports[0]
                date_str = most_recent.get("date", "")
                time_str = most_recent.get("time", "")
                if date_str and time_str:
                    return date_str, time_str
        
        # If not found in index, try to get from PDF file modification time
        if patient_name:
            pdf_file = self._find_report_file(patient_name, "")
            if pdf_file and os.path.exists(pdf_file):
                try:
                    mod_time = os.path.getmtime(pdf_file)
                    dt = datetime.datetime.fromtimestamp(mod_time)
                    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
                except:
                    pass
        
        # Fallback: use current time (should rarely happen)
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")

    def add_row(self, entry):
        """Append one row for a history entry."""
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Extract safe values
        date_str = entry.get("date", "")
        time_str = entry.get("time", "")
        org = entry.get("Org.", "")
        report_type = entry.get("report_type", "")
        doctor = entry.get("doctor", "")
        patient_name = entry.get("patient_name", "") or (
            (entry.get("first_name", "") + " " + entry.get("last_name", "")).strip()
        )
        age = str(entry.get("age", ""))
        gender = entry.get("gender", "")
        height = str(entry.get("height", ""))
        weight = str(entry.get("weight", ""))

        values = [
            date_str,
            time_str,
            org,
            doctor,
            patient_name,
            age,
            gender,
            height,
            weight,
            report_type,
        ]

        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, col, item)

        # Store report filename (if any) as row data for later open
        report_file = entry.get("report_file", "")
        self.table.setVerticalHeaderItem(row, QTableWidgetItem(""))
        self.table.setRowHeight(row, 24)
        # Use Qt.UserRole to store extra data on first column item
        if self.table.item(row, 0):
            self.table.item(row, 0).setData(Qt.UserRole, report_file)

    def on_row_double_clicked(self, row, column):
        """Handle double-click on a table row to open the report."""
        self.open_report_by_row(row)

    def open_selected_report(self):
        """Open the PDF report for the selected row, if available."""
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Open Report", "Please select a report row first.")
            return
        self.open_report_by_row(row)

    def open_report_by_row(self, row):
        """Open the PDF report for a specific row."""
        if row < 0 or row >= self.table.rowCount():
            return

        # First, try to get report file from stored data
        item = self.table.item(row, 0)
        if item:
            report_file = item.data(Qt.UserRole) or ""
            if report_file and os.path.exists(report_file):
                self._open_pdf_file(report_file)
                return

        # If no direct file path, try to find report based on patient details
        patient_name_item = self.table.item(row, 4)  # Patient Name column
        date_item = self.table.item(row, 0)  # Date column
        
        if not patient_name_item:
            QMessageBox.warning(
                self,
                "Open Report",
                "Could not find patient information for this entry."
            )
            return

        patient_name = patient_name_item.text().strip()
        date_str = date_item.text().strip() if date_item else ""

        # Try to find matching report file
        report_file = self._find_report_file(patient_name, date_str)
        
        if report_file and os.path.exists(report_file):
            self._open_pdf_file(report_file)
        else:
            QMessageBox.information(
                self,
                "Report Not Found",
                f"Could not find a PDF report for patient '{patient_name}'.\n\n"
                f"You can find all reports in the 'reports' folder."
            )

    def _find_report_file(self, patient_name, date_str=""):
        """Try to find a report file matching the patient name and optionally date."""
        reports_dir = os.path.join(BASE_DIR, "reports")
        if not os.path.exists(reports_dir):
            return None

        # Get all PDF files
        pdf_files = [f for f in os.listdir(reports_dir) if f.lower().endswith('.pdf')]
        
        # Try to find exact match by patient name in filename
        patient_name_clean = patient_name.replace(" ", "_").replace(",", "").upper()
        
        # First, try to find files with patient name
        for pdf_file in pdf_files:
            pdf_upper = pdf_file.upper()
            if patient_name_clean in pdf_upper:
                return os.path.join(reports_dir, pdf_file)
        
        # If date is provided, try to find by date pattern (YYYYMMDD)
        if date_str:
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                date_pattern = date_obj.strftime("%Y%m%d")
                
                for pdf_file in pdf_files:
                    if date_pattern in pdf_file:
                        return os.path.join(reports_dir, pdf_file)
            except:
                pass

        # Return most recent ECG_Report file if no match found
        ecg_reports = [f for f in pdf_files if f.startswith("ECG_Report_")]
        if ecg_reports:
            # Sort by modification time, most recent first
            ecg_reports.sort(key=lambda f: os.path.getmtime(os.path.join(reports_dir, f)), reverse=True)
            return os.path.join(reports_dir, ecg_reports[0])

        return None

    def _open_pdf_file(self, report_file):
        """Open a PDF file using the system's default PDF viewer."""
        try:
            if os.name == "nt":
                os.startfile(report_file)
            elif sys.platform == "darwin":
                os.system(f'open "{report_file}"')
            else:
                os.system(f'xdg-open "{report_file}"')
        except Exception as e:
            QMessageBox.critical(self, "Open Report", f"Failed to open report: {e}")

    def export_all_reports(self):
        """Export all saved reports from history to a user-selected directory."""
        # Ask user where to save the exported reports
        export_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Export All Reports",
            os.path.expanduser("~/Desktop"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not export_dir:
            return  # User cancelled
        
        try:
            reports_dir = os.path.join(BASE_DIR, "reports")
            if not os.path.exists(reports_dir):
                QMessageBox.warning(
                    self,
                    "Export Failed",
                    f"Reports directory not found: {reports_dir}"
                )
                return
            
            # Get all PDF files from reports directory
            pdf_files = [f for f in os.listdir(reports_dir) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                QMessageBox.information(
                    self,
                    "No Reports",
                    "No PDF reports found to export."
                )
                return
            
            # Create a subdirectory with timestamp for organized export
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_subdir = os.path.join(export_dir, f"ECG_Reports_Export_{timestamp}")
            os.makedirs(export_subdir, exist_ok=True)
            
            # Copy all PDF files
            copied_count = 0
            failed_count = 0
            
            for pdf_file in pdf_files:
                try:
                    src_path = os.path.join(reports_dir, pdf_file)
                    dst_path = os.path.join(export_subdir, pdf_file)
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                except Exception as e:
                    print(f" Failed to copy {pdf_file}: {e}")
                    failed_count += 1
            
            # Also export history data as CSV
            csv_path = os.path.join(export_subdir, "history_data.csv")
            try:
                with open(csv_path, "w", encoding="utf-8") as csv_file:
                    # Write header
                    csv_file.write("Date,Time,Report Type,Org.,Doctor,Patient Name,Age,Gender,Height (cm),Weight (kg)\n")
                    
                    # Write data from table
                    for row in range(self.table.rowCount()):
                        row_data = []
                        for col in range(self.table.columnCount()):
                            item = self.table.item(row, col)
                            value = item.text() if item else ""
                            # Escape commas and quotes in CSV
                            if "," in value or '"' in value:
                                value = '"' + value.replace('"', '""') + '"'
                            row_data.append(value)
                        csv_file.write(",".join(row_data) + "\n")
            except Exception as e:
                print(f" Failed to export CSV: {e}")
            
            # Show success message
            message = f"Export completed!\n\n"
            message += f"Location: {export_subdir}\n"
            message += f"PDFs copied: {copied_count}\n"
            if failed_count > 0:
                message += f"Failed: {failed_count}\n"
            message += f"\nHistory data exported as CSV."
            
            QMessageBox.information(
                self,
                "Export Successful",
                message
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export reports: {e}\n\nPlease check console for details."
            )
            import traceback
            traceback.print_exc()


def append_history_entry(patient_details, report_file_path, report_type="ECG"):
    """Append a new history entry when a report is generated."""
    print(f" append_history_entry called with patient_details={patient_details}, report_file_path={report_file_path}")
    
    # Load existing dedicated history file (rich entries)
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                entries = json.load(f)
        else:
            entries = []
            print(f" Creating new history file: {HISTORY_FILE}")
    except Exception as e:
        print(f" Error loading history file: {e}")
        entries = []

    if not isinstance(entries, list):
        entries = []

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    base = {
        "date": date_str,
        "time": time_str,
        "report_type": report_type,
        "report_file": os.path.abspath(report_file_path) if report_file_path else "",
    }
    if isinstance(patient_details, dict):
        base.update(patient_details)
        print(f" Merged patient details into base entry")
    else:
        print(f" patient_details is not a dict: {type(patient_details)}")

    entries.append(base)
    print(f" Added entry to history. Total entries: {len(entries)}")

    # Save rich history file
    try:
        # Ensure directory exists (HISTORY_FILE is in project root, so dirname should exist)
        history_dir = os.path.dirname(HISTORY_FILE)
        if history_dir and not os.path.exists(history_dir):
            os.makedirs(history_dir, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(entries, f, indent=2)
        print(f" Successfully saved history to {HISTORY_FILE}")
        print(f" Entry content: {json.dumps(base, indent=2)}")
    except Exception as e:
        # History is non-critical; just print warning
        print(f" Failed to append ECG history entry: {e}")
        import traceback
        traceback.print_exc()

