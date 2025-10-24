Privacy-Preserving Inference with CKKS (TenSEAL)

This project demonstrates how to perform privacy-preserving machine learning inference using the CKKS homomorphic encryption scheme. We use a simple logistic regression model and the TenSEAL library (Python interface to Microsoft SEAL) to show that a server can classify encrypted data without seeing the plaintext. The client encrypts data, the server computes on the ciphertext, and the client decrypts the result.

Installation and Environment

Python: Ensure you have Python 3.7 or higher installed.

Clone the repository (if applicable) or place the provided files in a directory.

Create a virtual environment (optional but recommended):

python3 -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate


Install dependencies:

pip install tenseal numpy scikit-learn


TenSEAL (for CKKS homomorphic encryption operations)

NumPy (for numerical arrays)

scikit-learn (for data generation and model)

Files

homomorphic_model.py: The main script. It generates synthetic data, trains a logistic regression model on plaintext, and then performs encrypted inference using CKKS. It prints the plaintext accuracy and the encrypted inference accuracy.

final_report.md: Detailed project report (introduction, methodology, experiments, etc.).

README.md: This file, with setup and usage instructions.

Usage (Demo)

Run the main script to see the comparison between plaintext and encrypted inference:

$ python homomorphic_model.py


You should see output similar to:

Plaintext Accuracy: 0.7800
Encrypted Accuracy: 0.7780


The exact numbers may vary slightly due to random data generation. The key point is that the encrypted accuracy closely matches the plaintext accuracy, demonstrating correct encrypted computation.

Expected Output

The script will output the test accuracy of the logistic regression model on plaintext data (e.g. ~0.78).

It will then perform encrypted inference on the same test data and output the accuracy after decryption (also ~0.78).

No sensitive data is printed or logged; only accuracy metrics are shown.

Demo Steps

Ensure all dependencies are installed (see above).

Run python homomorphic_model.py.

Observe the printed accuracies. Verify that the encrypted inference accuracy is essentially the same as the plaintext accuracy.

(Optional) Inspect the code to understand the encryption flow.

Environment Notes

This code runs on CPU. Performance (encryption/decryption speed) may be slow for large data, but it is sufficient for demonstration purposes.

TenSEAL requires C++ toolchain for installation; the pip install tenseal command downloads a pre-built wheel for common platforms. If you encounter issues, ensure your system meets TenSEAL’s requirements (e.g. a modern C++ compiler).

References

For background on CKKS and TenSEAL, see the TenSEAL paper and documentation:

A. Benaissa et al., “TenSEAL: A Library for Encrypted Tensor Operations Using Homomorphic Encryption”, ICLR 2021 workshop

Cheon et al., “Homomorphic Encryption for Arithmetic of Approximate Numbers (CKKS)”, ASIACRYPT 2017
arxiv.org
arxiv.o
