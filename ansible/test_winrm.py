import winrm

# Replace with your Windows host's IP address and your credentials
hostname = 'http://192.168.2.23:5985/wsman'
username = 'JANVI_RAI\\janvi'  # Replace with your actual Windows username
password = '2122'  # Replace with your actual Windows password

# Create a session to the Windows machine
session = winrm.Session(hostname, auth=(username, password))

# Run a command (for example, ipconfig)
result = session.run_cmd('ipconfig')

# Print the output of the command
print(result.std_out.decode())
print(result.std_err.decode())