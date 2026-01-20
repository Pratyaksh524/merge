# ECG Monitoring Software User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Running the Application](#running-the-application)
4. [User Authentication](#user-authentication)
    - [Sign Up](#sign-up)
    - [Sign In](#sign-in)
5. [Main Dashboard](#main-dashboard)
6. [12-Lead ECG Analysis](#12-lead-ecg-analysis)
7. [Expanded Lead View](#expanded-lead-view)
8. [Report Generation](#report-generation)
9. [Arrhythmia Detection](#arrhythmia-detection)
10. [Crash Logger and Diagnostics](#crash-logger-and-diagnostics)

## 1. Introduction
Welcome to the ECG Monitoring Software User Manual. This document provides a comprehensive guide to using the application, from initial setup to advanced features. The software is designed for real-time 12-lead ECG analysis, offering medical-grade signal filtering, live metric calculation, and detailed dashboard visualizations.

## 2. Installation
To get started with the ECG Monitoring Software, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/DivyansghDMK/macmodular.git
   cd macmodular
   ```

2. **Install Dependencies**:
   Make sure you have Python installed. Then, install the required libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Email Configuration (Optional)**:
   For crash reporting via email, you'll need to set up your Gmail credentials.
   - Copy the template file: `cp email_config_template.txt .env`
   - Edit the `.env` file and add your Gmail App Password.
   - For detailed instructions, refer to the `email_config_template.txt` file.

## 3. Running the Application
You can run the application using one of the following methods:

- **Option 1: Direct Python Execution (Recommended for macOS/Linux)**
  ```bash
  cd src
  python main.py
  ```

- **Option 2: Using Batch File (Windows)**
  ```bash
  launch_app.bat
  ```

- **Option 3: Using PowerShell Script (Windows)**
  ```powershell
  .\launch_app.ps1
  ```

## 4. User Authentication
The application requires user authentication to access the main features.

### Sign Up
If you are a new user, you need to create an account:
1. Open the application, and the login window will appear.
2. Click on the "Sign Up" tab.
3. Fill in the required details: Machine Serial ID, Full Name, Age, Gender, Address, Phone, Password, and Confirm Password.
4. Click the "Register" button.
5. Upon successful registration, you will be redirected to the login tab.

### Sign In
1. Open the application to the login window.
2. Enter your registered Full Name, Username, or Phone number, along with your password.
3. Click the "Login" button to access the dashboard.
4. There is also an admin shortcut available (`admin` / `adminsd`).

## 5. Main Dashboard
After signing in, you will be greeted by the main dashboard, which provides a real-time overview of the patient's vital signs.

- **Live Metrics**: The dashboard displays live calculations for:
  - Heart Rate (BPM)
  - PR Interval (ms)
  - QRS Duration (ms)
  - QTc Interval (ms)
  - QRS Axis (degrees)
  - ST Segment (mV)
- **12-Lead ECG Test**: Click the "ECG Lead Test 12" button to open the detailed 12-lead analysis page.
- **Recent Reports**: The panel on the right lists the last 10 generated PDF reports, which you can open directly.

## 6. 12-Lead ECG Analysis
This view provides a comprehensive analysis of the 12-lead ECG signals.

- **Real-time Waveforms**: Displays all 12 lead waveforms simultaneously.
- **Expanded Lead View**: Click on any of the lead waveforms to open an expanded, detailed view for that specific lead.
- **Arrhythmia Detection**: The system continuously monitors for and displays any detected cardiac arrhythmias.

## 7. Expanded Lead View
This view allows for a detailed analysis of a single ECG lead.

- **PQRST Labeling**: The software automatically identifies and labels the P, Q, R, S, and T waves on the ECG waveform.
- **Detailed Metrics**: Provides specific measurements for the selected lead.

## 8. Report Generation
The application can generate comprehensive ECG reports in PDF format.

1. From the main dashboard or the 12-lead analysis view, click the "Generate Report" button.
2. The report will be saved in two locations:
   - A location of your choice (e.g., your Downloads folder).
   - The `reports/` directory within the application folder, for historical access.
3. The "Recent Reports" panel on the dashboard will be updated with the newly generated report.

## 9. Arrhythmia Detection
The software includes a real-time arrhythmia detection feature. It automatically identifies and flags various types of cardiac arrhythmias, which are displayed on the 12-lead analysis screen.

## 10. Crash Logger and Diagnostics
The application includes a hidden diagnostic system for troubleshooting and crash reporting.

### Accessing the Crash Logger
1. **Triple-click** the heart rate metric on the main dashboard.
2. A diagnostic dialog will open, showing:
   - Session statistics (duration, errors, crashes, memory usage)
   - Email configuration status
   - Crash logs and error reports
3. From this dialog, you can send reports via email or clear the logs.





