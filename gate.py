import socket
import RPi.GPIO as GPIO
import time

# GPIO setup
RELAY_PIN = 17  # Replace with the GPIO pin connected to the relay
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.HIGH)  # Ensure relay is off initially (depends on relay type)

# Socket setup
HOST = '192.168.1.88'  # Replace with the IP address of the server
PORT = 5005

# Function to control the relay
def control_relay(command):
    if command == b'RELAY_ON':
        GPIO.output(RELAY_PIN, GPIO.LOW)  # Turn on the relay
        return b'RELAY_ON_ACK'
    elif command == b'RELAY_OFF':
        GPIO.output(RELAY_PIN, GPIO.HIGH)  # Turn off the relay
        return b'RELAY_OFF_ACK'
    else:
        return b'UNKNOWN_COMMAND'

# Start the server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Server listening on {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                response = control_relay(data)
                conn.sendall(response)

# Clean up GPIO on exit
GPIO.cleanup()


