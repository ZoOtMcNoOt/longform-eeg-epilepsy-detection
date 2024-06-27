import pyopencl as cl

# List all available OpenCL platforms and devices
platforms = cl.get_platforms()
for platform in platforms:
    print(f"Platform: {platform.name}")
    devices = platform.get_devices()
    for device in devices:
        print(f"  Device: {device.name}, Type: {cl.device_type.to_string(device.type)}")
