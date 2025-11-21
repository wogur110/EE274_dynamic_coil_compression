import os
import sys
import subprocess

def run_script(script_name):
    print(f"\nRunning {script_name}...")
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
    except Exception as e:
        print(f"Failed to launch {script_name}: {e}")

def main():
    # Create results directory
    os.makedirs("results", exist_ok=True)

    # 1. Generate Reference
    run_script(os.path.join("scripts", "refer0_generate_reference.py"))
    
    # 2. JPEG
    run_script(os.path.join("scripts", "refer1a_jpeg_compression.py"))
    
    # 3. DCT Compression (New Phase 1B variant)
    run_script(os.path.join("scripts", "refer1b_DCT_compression.py"))
    
    # 4. FFT Compression (Undersampling / VD Poisson)
    run_script(os.path.join("scripts", "refer1c_FFT_compression.py"))
    
    # 5. Uniform Coil Compression (PCA)
    run_script(os.path.join("scripts", "uniform_coil_compression.py"))
    
    # # 6. Dynamic Coil Compression
    # run_script(os.path.join("scripts", "dynamic_coil_compression.py"))
    
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
