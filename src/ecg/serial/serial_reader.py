"""Serial communication classes for ECG hardware"""
import time
from typing import List, Dict
from PyQt5.QtWidgets import QMessageBox

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    print(" Serial module not available - ECG hardware features disabled")
    SERIAL_AVAILABLE = False
    # Create dummy serial classes
    class Serial:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
        def readline(self): return b''
        def read(self, size): return b''
        def write(self, data): pass
        def reset_input_buffer(self): pass
        def reset_output_buffer(self): pass
        @property
        def is_open(self): return False
    class SerialException(Exception): pass
    serial = type('Serial', (), {'Serial': Serial, 'SerialException': SerialException})()
    class MockComports:
        @staticmethod
        def comports(*args, **kwargs):
            return []
    serial.tools = type('Tools', (), {'list_ports': MockComports()})()

from utils.crash_logger import get_crash_logger
from .packet_parser import parse_packet, PACKET_SIZE, START_BYTE, END_BYTE


class SerialStreamReader:
    """Packet-based serial reader for ECG data - NEW IMPLEMENTATION"""
    
    def __init__(self, port: str, baudrate: int, timeout: float = 0.1):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial is required for serial capture. pip install pyserial")
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self.buf = bytearray()
        self.running = False
        self.data_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time = 0
        self.crash_logger = get_crash_logger()
        self.user_details = {}  # For error reporting compatibility
        # Packet loss tracking
        self.start_time = time.time()
        self.last_packet_time = time.time()
        self.total_packets_expected = 0
        self.total_packets_lost = 0
        self.packet_loss_percent = 0.0
        # Sequence-based packet loss detection
        self._last_packet_counter = None
        self._total_sequence_lost = 0
        self._packet_loss_warnings = 0
        print(f" SerialStreamReader initialized: Port={port}, Baud={baudrate}")

    def close(self) -> None:
        """Close serial connection"""
        try:
            # Stop data acquisition first
            self.running = False
            
            # Flush any remaining data
            if hasattr(self.ser, 'reset_input_buffer'):
                try:
                    self.ser.reset_input_buffer()
                except Exception:
                    pass
            
            if hasattr(self.ser, 'reset_output_buffer'):
                try:
                    self.ser.reset_output_buffer()
                except Exception:
                    pass
            
            # Close the serial port
            if self.ser and self.ser.is_open:
                self.ser.close()
                print(" Serial port closed and released")
            
            # Clear buffer
            self.buf.clear()
        except Exception as e:
            print(f" Error closing serial connection: {e}")

    def start(self):
        """Start data acquisition"""
        print(" Starting packet-based ECG data acquisition...")
        self.ser.reset_input_buffer()
        self.buf.clear()
        self.running = True
        # Initialize packet loss tracking
        self.start_time = time.time()
        self.last_packet_time = time.time()
        self.data_count = 0
        self.total_packets_expected = 0
        self.total_packets_lost = 0
        self.packet_loss_percent = 0.0
        print(" Packet-based ECG device started - waiting for data packets...")

    def stop(self):
        """Stop data acquisition"""
        print(" Stopping packet-based ECG data acquisition...")
        self.running = False
        # Final packet loss statistics
        if hasattr(self, 'start_time') and self.start_time > 0:
            elapsed_time = time.time() - self.start_time
            expected_packets = int(500 * elapsed_time)  # 500 Hz
            total_lost = max(0, expected_packets - self.data_count)
            loss_percent = (total_lost / expected_packets * 100) if expected_packets > 0 else 0
            print(f" Total data packets received: {self.data_count}")
            if total_lost > 0:
                print(f" Packet loss summary: {total_lost}/{expected_packets} packets lost ({loss_percent:.2f}% loss)")
            else:
                print(f" No packet loss detected - all {expected_packets} expected packets received")
            
            # Report sequence-based packet loss
            if hasattr(self, '_total_sequence_lost') and self._total_sequence_lost > 0:
                print(f" Sequence-based packet loss: {self._total_sequence_lost} packets lost (detected via counter gaps)")
            else:
                print(f" No sequence gaps detected - perfect packet continuity!")
        else:
            print(f" Total data packets received: {self.data_count}")
            if hasattr(self, '_total_sequence_lost') and self._total_sequence_lost > 0:
                print(f" Sequence-based packet loss: {self._total_sequence_lost} packets lost")

    def read_packets(self, max_packets: int = 100) -> List[Dict[str, int]]:
        """Read and parse ECG packets from serial stream
        
        At 500 Hz, hardware sends 500 packets/second = ~16.67 packets per 33ms timer interval.
        We read up to max_packets to prevent buffer overflow and packet loss.
        """
        if not self.running:
            return []
            
        out: List[Dict[str, int]] = []
        
        try:
            # Read larger chunks to prevent buffer overflow at 500 Hz
            # At 500 Hz with 22-byte packets = 11,000 bytes/second
            # Read up to 4096 bytes per call to catch up quickly
            chunk = self.ser.read(4096)  # Increased from default (usually 1024)
            if chunk:
                self.buf.extend(chunk)

            # Extract packets - process ALL available packets to prevent buffer overflow
            # At 500 Hz, we need to process packets quickly to avoid accumulation
            # Read ALL packets in buffer, not just max_packets, to prevent overflow
            max_iterations = max_packets * 3  # Allow catching up if we fell behind
            iteration = 0
            while iteration < max_iterations:
                iteration += 1
                start_idx = self.buf.find(bytes([START_BYTE]))
                if start_idx == -1:
                    # No start byte found - clear buffer if it's getting too large (>100KB)
                    if len(self.buf) > 100000:
                        print(f" Serial buffer overflow risk: {len(self.buf)} bytes, clearing buffer")
                        self.buf.clear()
                    break
                if start_idx > 10000:
                    # Skip too much garbage data before start byte
                    del self.buf[:start_idx]
                    continue
                if len(self.buf) - start_idx < PACKET_SIZE:
                    # Not enough data for a complete packet - keep what we have
                    if start_idx > 0:
                        del self.buf[:start_idx]
                    break
                    
                candidate = bytes(self.buf[start_idx : start_idx + PACKET_SIZE])
                del self.buf[: start_idx + PACKET_SIZE]

                if candidate[-1] != END_BYTE:
                    continue

                parsed = parse_packet(candidate)
                if parsed:
                    self.data_count += 1
                    self.last_packet_time = time.time()
                    
                    # Extract packet counter for sequence tracking
                    packet_counter = candidate[1] & 0x3F  # Counter is in lower 6 bits (0-63)
                    
                    # Detect packet loss by checking sequence continuity
                    if self._last_packet_counter is not None:
                        expected_counter = (self._last_packet_counter + 1) % 64
                        if packet_counter != expected_counter:
                            # Calculate how many packets were lost
                            if packet_counter > expected_counter:
                                lost = packet_counter - expected_counter
                            else:
                                # Wrapped around (e.g., 63 -> 0)
                                lost = (64 - expected_counter) + packet_counter
                            
                            if lost > 0:
                                self._total_sequence_lost += lost
                                self._packet_loss_warnings += 1
                                # Only warn every 10th occurrence to avoid spam
                                if self._packet_loss_warnings % 10 == 1:
                                    print(f" PACKET LOSS DETECTED: {lost} packet(s) dropped! "
                                          f"Expected counter: {expected_counter}, Got: {packet_counter}. "
                                          f"Total sequence lost: {self._total_sequence_lost}")
                    
                    self._last_packet_counter = packet_counter
                    
                    # Only log every 100th packet to reduce console spam
                    if self.data_count % 100 == 0:
                        loss_info = f" (Sequence lost: {self._total_sequence_lost})" if self._total_sequence_lost > 0 else ""
                        print(f" [Packet #{self.data_count}, Counter: {packet_counter}]{loss_info} Received valid packet with {len(parsed)} leads")
                    out.append(parsed)
            
            # Warn if buffer is accumulating too much data (indicates we're falling behind)
            if len(self.buf) > 50000:  # >50KB buffer indicates we're not reading fast enough
                print(f" Serial buffer accumulation: {len(self.buf)} bytes - may indicate packet loss")
            
            # Update packet loss statistics
            if self.running and self.data_count > 0:
                elapsed_time = time.time() - self.start_time
                expected_packets = int(500 * elapsed_time)  # 500 Hz = 500 packets/second
                self.total_packets_expected = expected_packets
                self.total_packets_lost = max(0, expected_packets - self.data_count)
                if expected_packets > 0:
                    self.packet_loss_percent = (self.total_packets_lost / expected_packets) * 100
                    
        except Exception as e:
            self.error_count += 1
            self.consecutive_errors += 1
            error_msg = f"Packet parsing error: {e}"
            print(f" {error_msg}")
            self.crash_logger.log_error(
                message=error_msg,
                exception=e,
                category="SERIAL_ERROR"
            )
            
            # If device is disconnected (Errno 6) or too many consecutive errors, stop
            if "Device not configured" in str(e) or "[Errno 6]" in str(e) or self.consecutive_errors > 20:
                print(" Critical serial error - stopping acquisition")
                self.running = False
            
        return out

    def _handle_serial_error(self, error):
        """Handle serial communication errors"""
        current_time = time.time()
        self.error_count += 1
        self.consecutive_errors += 1
        
        error_msg = f"Serial communication error: {error}"
        print(f" {error_msg}")
        
        self.crash_logger.log_error(
            message=error_msg,
            exception=error,
            category="SERIAL_ERROR"
        )
        
        if self.consecutive_errors >= 5 and (current_time - self.last_error_time) > 10:
            self.last_error_time = current_time
            self.consecutive_errors = 0


class SerialECGReader:
    """Legacy serial reader for line-based ECG data"""
    def __init__(self, port, baudrate):
        if not SERIAL_AVAILABLE:
            raise ImportError("Serial module not available - cannot create ECG reader")
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.running = False
        self.data_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time = 0
        self.crash_logger = get_crash_logger()
        print(f" SerialECGReader initialized: Port={port}, Baud={baudrate}")

    def start(self):
        print(" Starting ECG data acquisition...")
        self.ser.reset_input_buffer()
        self.ser.write(b'1\r\n')
        time.sleep(0.5)
        self.running = True
        print(" ECG device started - waiting for data...")

    def stop(self):
        print(" Stopping ECG data acquisition...")
        self.ser.write(b'0\r\n')
        self.running = False
        print(f" Total data packets received: {self.data_count}")

    def read_value(self):
        if not self.running:
            return None
        try:
            line_raw = self.ser.readline()
            line_data = line_raw.decode('utf-8', errors='replace').strip()

            if line_data:
                self.data_count += 1
                # Print detailed data information
                print(f" [Packet #{self.data_count}] Raw data: '{line_data}' (Length: {len(line_data)})")
                
                # Parse and display ECG value
                if line_data.isdigit():
                    ecg_value = int(line_data[-3:])
                    print(f" ECG Value: {ecg_value} mV")
                    return ecg_value
                else:
                    # Try to parse as multiple values (8-channel data)
                    try:
                        # Clean the line data - remove any non-numeric characters except spaces and minus signs
                        import re
                        cleaned_line = re.sub(r'[^\d\s\-]', ' ', line_data)
                        values = [int(x) for x in cleaned_line.split() if x.strip() and x.replace('-', '').isdigit()]
                        
                        if len(values) >= 8:
                            print(f" 8-Channel ECG Data: {values}")
                            return values  # Return the list of 8 values
                        elif len(values) == 1:
                            print(f" Single ECG Value: {values[0]} mV")
                            return values[0]
                        elif len(values) > 0:
                            print(f" Unexpected number of values: {len(values)} (expected 8)")
                        else:
                            return None
                    except Exception as e:
                        print(f" Error parsing ECG data: {e}")
                        return None
            else:
                print("⏳ No data received (timeout)")
                
        except Exception as e:
            self._handle_serial_error(e)
        return None

    def close(self):
        print(" Closing serial connection...")
        self.ser.close()
        print(" Serial connection closed")

    def _handle_serial_error(self, error):
        """Handle serial communication errors with alert and logging"""
        current_time = time.time()
        self.error_count += 1
        self.consecutive_errors += 1
        
        # Log the error
        error_msg = f"Serial communication error: {error}"
        print(f" {error_msg}")
        
        # Log to crash logger
        self.crash_logger.log_error(
            message=error_msg,
            exception=error,
            category="SERIAL_ERROR"
        )
        
        # Show alert if consecutive errors exceed threshold
        if self.consecutive_errors >= 5 and (current_time - self.last_error_time) > 10:
            self._show_serial_error_alert(error)
            self.last_error_time = current_time
            self.consecutive_errors = 0  # Reset counter after showing alert
    
    def _show_serial_error_alert(self, error):
        """Show alert dialog for serial communication errors"""
        try:
            # Get user details from main application
            user_details = getattr(self, 'user_details', {})
            username = user_details.get('full_name', 'Unknown User')
            phone = user_details.get('phone', 'N/A')
            email = user_details.get('email', 'N/A')
            serial_id = user_details.get('serial_id', 'N/A')
            
            # Create detailed error message
            error_details = f"""
Serial Communication Error Detected!

Error: {str(error)}
User: {username}
Phone: {phone}
Email: {email}
Serial ID: {serial_id}
Machine Serial: {self.crash_logger.machine_serial_id or 'N/A'}
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

This error has been logged and an email notification will be sent to the support team.
            """
            
            # Show alert dialog
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Serial Communication Error")
            msg_box.setText("ECG Device Connection Lost")
            msg_box.setDetailedText(error_details)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
            
            # Send email notification
            self._send_error_email(error, user_details)
            
        except Exception as e:
            print(f" Error showing serial error alert: {e}")
    
    def _send_error_email(self, error, user_details):
        """Send email notification for serial errors"""
        try:
            # Create error data for email
            error_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error_type': 'Serial Communication Error',
                'error_message': str(error),
                'user_details': user_details,
                'machine_serial': self.crash_logger.machine_serial_id or 'N/A',
                'consecutive_errors': self.consecutive_errors,
                'total_errors': self.error_count
            }
            
            # Send email using crash logger
            self.crash_logger._send_crash_email(error_data)
            print(" Serial error email notification sent")
            
        except Exception as e:
            print(f" Error sending serial error email: {e}")
