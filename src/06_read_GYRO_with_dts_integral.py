import smbus2
import time

# I2C address of the WT901 sensor (default is 0x50)
DEVICE_ADDRESS = 0x50
# Gyro Z register
GYRO_Z_REGISTER = 0x39  # Example register for Z-axis gyroscope

# Initialize the I2C bus
bus = smbus2.SMBus(1)

# Variable to store the yaw angle
yaw_angle = 0

# Time tracking for integration
previous_time = time.time()

def read_gyro_z():
    # Read two bytes from the gyro Z register
    data = bus.read_i2c_block_data(DEVICE_ADDRESS, GYRO_Z_REGISTER, 2)
    # Combine the two bytes into a 16-bit value
    gyro_z_raw = data[1] << 8 | data[0]
    
    # Convert the raw gyroscope value to angular velocity in degrees per second
    # (You might need to adjust the scale depending on your gyro's sensitivity)
    if gyro_z_raw > 32767:
        gyro_z_raw -= 65536
    gyro_z = gyro_z_raw / 32768.0 * 2000  # Assuming +/- 2000 dps range, adjust if needed
    return gyro_z





try:
    while True:
        # Read the current gyroscope value (Z-axis)
        gyro_z = read_gyro_z()
        if gyro_z > 0:
            gyro_z = gyro_z * 1.008

        # Get the current time
        current_time = time.time()

        # Calculate time difference (delta t)
        delta_t = current_time - previous_time

        # Update the yaw angle using gyro data (angular velocity * time)
        yaw_angle += gyro_z * delta_t

        # Update previous time for the next iteration
        previous_time = current_time

        # Print the current yaw angle
        print("Yaw Angle: {:.2f} degrees".format(yaw_angle))

        # Wait for a short period before the next reading
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Program interrupted")