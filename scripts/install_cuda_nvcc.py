#!/usr/bin/env python3

import subprocess
import os
import sys
import platform

def run_command(command, return_output=False):
    """
    Runs a shell command and exits if the command fails.
    """
    try:
        if return_output:
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
            return output.strip()
        else:
            subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while executing: {command}")
        if return_output:
            print(f"Command output:\n{e.output}")
        sys.exit(1)

def check_cuda_capable_gpu():
    """
    Verifies if the system has a CUDA-capable GPU.
    """
    print("Checking for CUDA-capable GPU...")
    output = run_command("lspci | grep -i nvidia", return_output=True)
    if output:
        print("CUDA-capable GPU detected:")
        print(output)
    else:
        print("No NVIDIA GPU detected. Updating PCI hardware database and rechecking...")
        run_command("sudo update-pciids")
        output = run_command("lspci | grep -i nvidia", return_output=True)
        if output:
            print("CUDA-capable GPU detected after updating PCI IDs:")
            print(output)
        else:
            print("No NVIDIA GPU detected. Please ensure you have a CUDA-capable GPU installed.")
            print("Refer to https://developer.nvidia.com/cuda-gpus for a list of CUDA-capable GPUs.")
            sys.exit(1)

def check_linux_version():
    """
    Verifies the system is running a supported version of Linux.
    """
    print("Checking Linux distribution and version...")
    arch = run_command("uname -m", return_output=True)
    release_info = run_command("cat /etc/*release", return_output=True)
    print(f"System architecture: {arch}")
    print(f"Release information:\n{release_info}")
    # You can add additional checks here to verify if the Linux distribution is supported.

def check_gcc_installed():
    """
    Verifies that gcc is installed on the system.
    """
    print("Checking if gcc is installed...")
    try:
        gcc_version = run_command("gcc --version", return_output=True)
        print(f"gcc is installed:\n{gcc_version}")
    except SystemExit:
        print("gcc is not installed. Please install gcc before proceeding.")
        sys.exit(1)

def check_kernel_headers():
    """
    Verifies that the correct kernel headers and development packages are installed.
    """
    print("Checking for kernel headers and development packages...")
    kernel_version = run_command("uname -r", return_output=True)
    print(f"Running kernel version: {kernel_version}")

    distro = get_distro_info()
    distro_name = distro['name']
    package_manager = get_package_manager(distro_name)

    if package_manager == 'apt':
        headers_installed = run_command(f"dpkg -l | grep linux-headers-{kernel_version}", return_output=True)
        if headers_installed:
            print(f"Kernel headers for version {kernel_version} are installed.")
        else:
            print(f"Kernel headers for version {kernel_version} are not installed.")
            print("Installing kernel headers...")
            run_command(f"sudo apt-get install -y linux-headers-$(uname -r)")
    elif package_manager == 'dnf' or package_manager == 'yum':
        headers_installed = run_command(f"rpm -qa | grep kernel-devel-{kernel_version}", return_output=True)
        if headers_installed:
            print(f"Kernel development packages for version {kernel_version} are installed.")
        else:
            print(f"Kernel development packages for version {kernel_version} are not installed.")
            print("Installing kernel development packages...")
            run_command(f"sudo {package_manager} install -y kernel-devel-$(uname -r)")
    else:
        print("Unsupported package manager. Please install the kernel headers manually.")
        sys.exit(1)

def get_distro_info():
    """
    Returns the distribution name and version.
    """
    try:
        import distro
        distro_name = distro.id()
        distro_version = distro.version()
    except ImportError:
        # For older versions of Python where 'distro' is not available
        distro_name, distro_version, _ = platform.linux_distribution()

    distro_name_lower = distro_name.lower()
    return {'name': distro_name_lower, 'version': distro_version}

def get_package_manager(distro_name):
    """
    Determines the package manager based on the distribution.
    """
    if 'ubuntu' in distro_name or 'debian' in distro_name:
        return 'apt'
    elif 'fedora' in distro_name or 'centos' in distro_name or 'rhel' in distro_name:
        return 'dnf' if 'fedora' in distro_name or 'rhel' in distro_name else 'yum'
    else:
        return None

def check_os_version():
    """
    Checks the operating system and version.
    """
    distro = get_distro_info()
    distro_name = distro['name']
    distro_version = distro['version']

    print(f"Detected operating system: {distro_name.capitalize()} {distro_version}")

    return distro

def install_cuda_ubuntu():
    """
    Installs CUDA Toolkit and NVCC compiler on Ubuntu 24.04.
    """
    print("Installing CUDA Toolkit for Ubuntu 24.04...")

    # Update package lists
    run_command("sudo apt-get update")

    # Install prerequisite packages
    run_command("sudo apt-get install -y build-essential dkms")

    # Download the NVIDIA CUDA GPG key
    run_command("wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub")

    # Add the GPG key
    run_command("sudo mv 3bf863cc.pub /usr/share/keyrings/cuda-archive-keyring.gpg")

    # Add the CUDA repository
    run_command('echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list')

    # Update package lists again
    run_command("sudo apt-get update")

    # Install CUDA
    run_command("sudo apt-get -y install cuda-toolkit-12-6")

    # Set environment variables
    cuda_path = "/usr/local/cuda"
    set_cuda_environment(cuda_path)

def set_cuda_environment(cuda_path):
    """
    Sets the CUDA environment variables.
    """
    print(f"Setting up CUDA environment variables for {cuda_path}")
    bashrc_path = os.path.expanduser("~/.bashrc")
    with open(bashrc_path, "a") as bashrc:
        bashrc.write(f"\n# CUDA environment variables\n")
        bashrc.write(f"export PATH={cuda_path}/bin:$PATH\n")
        bashrc.write(f"export LD_LIBRARY_PATH={cuda_path}/lib64:$LD_LIBRARY_PATH\n")

    # Apply the changes in the current session
    os.environ['PATH'] = f"{cuda_path}/bin:" + os.environ['PATH']
    os.environ['LD_LIBRARY_PATH'] = f"{cuda_path}/lib64:" + os.environ.get('LD_LIBRARY_PATH', '')

    print("CUDA environment variables have been set.")

def verify_installation():
    """
    Verifies that CUDA and NVCC are installed correctly.
    """
    print("Verifying CUDA installation...")
    try:
        nvcc_version = run_command("nvcc --version", return_output=True)
        print("NVCC version:")
        print(nvcc_version)
    except SystemExit:
        print("Failed to verify NVCC installation.")
        sys.exit(1)

    try:
        nvidia_smi = run_command("nvidia-smi", return_output=True)
        print("NVIDIA SMI output:")
        print(nvidia_smi)
    except SystemExit:
        print("Failed to verify NVIDIA driver installation.")
        sys.exit(1)

    print("CUDA Toolkit installation completed successfully.")

def display_install_instructions(distro_name, distro_version):
    """
    Displays the installation instructions for the given distribution.
    """
    if 'fedora' in distro_name and distro_version == '39':
        print("\nDownload Installer for Linux Fedora 39 x86_64")
        print("The base installer is available for download below.\n")
        print("CUDA Toolkit Installer")
        print("Installation Instructions:")
        print("sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo")
        print("sudo dnf clean all")
        print("sudo dnf -y install cuda-toolkit-12-6")
        print("Additional installation options are detailed here.")
        print("Driver Installer")
        print("NVIDIA Driver Instructions (choose one option)")
        print("To install the open kernel module flavor:")
        print("sudo dnf -y module install nvidia-driver:open-dkms")
        print("To install the legacy kernel module flavor:")
        print("sudo dnf -y module install nvidia-driver:latest-dkms")
    elif 'debian' in distro_name and distro_version.startswith('12'):
        print("\nDownload Installer for Linux Debian 12 x86_64")
        print("The base installer is available for download below.\n")
        print("CUDA Toolkit Installer")
        print("Installation Instructions:")
        print("wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb")
        print("sudo dpkg -i cuda-keyring_1.1-1_all.deb")
        print("sudo add-apt-repository contrib")
        print("sudo apt-get update")
        print("sudo apt-get -y install cuda-toolkit-12-6")
        print("Additional installation options are detailed here.")
        print("Driver Installer")
        print("NVIDIA Driver Instructions (choose one option)")
        print("To install the open kernel module flavor:")
        print("sudo apt-get install -y nvidia-open")
        print("To install the legacy kernel module flavor:")
        print("sudo apt-get install -y cuda-drivers")
    elif 'rhel' in distro_name and distro_version.startswith('9'):
        print("\nDownload Installer for Linux RHEL 9 x86_64")
        print("The base installer is available for download below.\n")
        print("CUDA Toolkit Installer")
        print("Installation Instructions:")
        print("sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo")
        print("sudo dnf clean all")
        print("sudo dnf -y install cuda-toolkit-12-6")
        print("Additional installation options are detailed here.")
        print("Driver Installer")
        print("NVIDIA Driver Instructions (choose one option)")
        print("To install the open kernel module flavor:")
        print("sudo dnf -y module install nvidia-driver:open-dkms")
        print("To install the legacy kernel module flavor:")
        print("sudo dnf -y module install nvidia-driver:latest-dkms")
    else:
        print("\nThis installation script does not support your operating system automatically.")
        print("Please install the following NVIDIA packages manually:")
        print(" - NVIDIA Driver")
        print(" - CUDA Toolkit")
        print(" - NVCC Compiler")
        print("\nRefer to the NVIDIA installation instructions for your distribution:")
        print("https://developer.nvidia.com/cuda-downloads?target_os=Linux")
    sys.exit(1)

def main():
    # Check for CUDA-capable GPU
    check_cuda_capable_gpu()

    # Verify the system has gcc installed
    check_gcc_installed()

    # Verify the system has the correct kernel headers and development packages
    check_kernel_headers()

    # Check the operating system and version
    distro = check_os_version()
    distro_name = distro['name']
    distro_version = distro['version']

    if 'ubuntu' in distro_name and distro_version == '24.04':
        # Install CUDA Toolkit for Ubuntu 24.04
        install_cuda_ubuntu()
        # Verify installation
        verify_installation()
    else:
        # Display installation instructions for other supported distributions
        display_install_instructions(distro_name, distro_version)

if __name__ == "__main__":
    main()
