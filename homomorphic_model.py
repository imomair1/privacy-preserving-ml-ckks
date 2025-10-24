"""
homomorphic_model.py

Demonstration of privacy-preserving inference using the CKKS homomorphic encryption
scheme with the TenSEAL library. This script trains a simple machine learning model
(plaintext), then performs inference on encrypted data using the CKKS scheme, and
compares the accuracy of encrypted inference to plaintext inference.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import tenseal as ts

def main():
    # ------------------------------
    # Step 1: Data preparation and model training (plaintext)
    # ------------------------------
    # Create a synthetic binary classification dataset
    X, y = make_classification(n_samples=1000, n_features=10,
                               n_informative=5, random_state=1)
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Train a logistic regression model on plaintext data
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    # Evaluate plaintext model accuracy
    plain_accuracy = model.score(X_test, y_test)
    print(f"Plaintext Accuracy: {plain_accuracy:.4f}")

    # Extract model weights and bias for inference
    weights = model.coef_[0].tolist()      # List of weight coefficients
    bias = float(model.intercept_[0])      # Bias term

    # ------------------------------
    # Step 2: Setup TenSEAL CKKS context (encryption parameters)
    # ------------------------------
    # Configure CKKS parameters: polynomial modulus degree and coefficient modulus sizes
    # These parameters affect security and precision.
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()         # Needed for vector rotation (dot product)
    context.global_scale = 2**40          # Scaling factor for fixed-point precision

    # ------------------------------
    # Step 3: Perform encrypted inference (CKKS)
    # ------------------------------
    encrypted_preds = []
    for sample in X_test:
        # Encrypt the feature vector
        enc_sample = ts.ckks_vector(context, sample.tolist())
        # Compute encrypted dot product with model weights
        enc_result = enc_sample.dot(weights)  # Encrypted dot product (weights * features)
        # Add the bias (as plaintext scalar) to the encrypted result
        enc_result = enc_result + bias
        # Decrypt the result to obtain the predicted logit value
        result = enc_result.decrypt()[0]      # CKKS decrypt returns a list
        # Determine class (threshold at 0 for logistic model)
        pred = 1 if result > 0 else 0
        encrypted_preds.append(pred)

    encrypted_preds = np.array(encrypted_preds)
    encrypted_accuracy = np.mean(encrypted_preds == y_test)
    print(f"Encrypted Accuracy: {encrypted_accuracy:.4f}")

if __name__ == "__main__":
    main()
