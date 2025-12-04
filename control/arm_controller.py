# control/arm_controller.py

import socket
import json
import time
from config import SERVER_IP, SERVER_PORT, SEND_INTERVAL

class CorrectArmController:
    """Correct direction robotic arm controller"""
    def __init__(self, server_ip=SERVER_IP, server_port=SERVER_PORT):
        self.server_ip = server_ip
        self.server_port = server_port
        self.sock = None
        self.connected = False
        self.last_send_time = 0
        self.last_command = None
        
        self.connect_to_server()
    
    def connect_to_server(self):
        """Connect to Raspberry Pi server"""
        try:
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
                
            print(f"🔌 Trying to connect to {self.server_ip}:{self.server_port}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3.0)
            self.sock.connect((self.server_ip, self.server_port))
            self.sock.settimeout(5.0)
            self.connected = True
            print(f"✅ Connected successfully")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            self.connected = False
            return False
    
    def send_command(self, x, y, z):
        """Send command to robotic arm"""
        current_time = time.time()
        
        # Check send interval to avoid sending too frequently
        if current_time - self.last_send_time < SEND_INTERVAL:
            return True
            
        if not self.connected:
            if not self.connect_to_server():
                return False
        
        # Command format: x, y, z coordinates
        command = {'x': round(x, 1), 'y': round(y, 1), 'z': round(z, 1)}
        
        try:
            command_str = json.dumps(command)
            self.sock.sendall(command_str.encode('utf-8'))
            # print(f"🎯 Sent: {command_str}")
            
            # Try to receive response
            self.sock.settimeout(0.5)
            try:
                response = self.sock.recv(128)
                if response:
                    response_str = response.decode().strip()
                    # print(f"📥 Response: {response_str}")
            except socket.timeout:
                pass
            
            self.sock.settimeout(5.0)
            self.last_send_time = current_time
            self.last_command = (x, y, z)
            return True
            
        except socket.timeout:
            # print("⚠️ Send timeout")
            self.connected = False
            return False
        except Exception as e:
            # print(f"❌ Send failed: {str(e)}")
            self.connected = False
            return False
    
    def stop(self):
        """Stop the controller"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass