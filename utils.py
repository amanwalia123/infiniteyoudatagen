import os
from typing import Literal
import socket
import requests
import netifaces

def get_server_name() -> Literal["SPACE", "MLP", "lambda"]:
    no_proxy = os.environ.get("no_proxy", "")
    server = None

    if any(substring in no_proxy for substring in [".sjc."]):
        server = "MLP"
    elif any(substring in no_proxy for substring in [".n6.", ".n9.", ".n8."]):
        server = "SPACE"
        # if os.path.exists("/datalake"):
        #     server = "datalake"
    else:
        server = "lambda"

    return server

def get_ip_address():
    """Get the IPv4 address for a given network interface."""
    server_name = get_server_name()
    
    if server_name == "lambda":
        interface_name = "enp36s0f0"
    elif server_name == "MLP" or server_name == "SPACE":
        interface_name = "eth0"
    else:  # SPACE
        raise ValueError("Unknown server name")
        
    try:
        # Get the address information for the specified interface
        addresses = netifaces.ifaddresses(interface_name)

        # Look for an IPv4 address (AF_INET)
        if netifaces.AF_INET in addresses:
            # Return the address from the first IPv4 entry
            return addresses[netifaces.AF_INET][0]['addr']
        else:
            return "No IPv4 address found for this interface."
    except ValueError:
        return "Error: Interface not found."
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    print(get_server_name())
    print(get_ip_address())