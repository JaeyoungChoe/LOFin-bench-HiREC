import json
import base64
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import argparse
from typing import Tuple
import sys

def generate_key(password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
    """Generate an encryption key from a password."""
    if salt is None:
        salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key, salt

def encrypt_file(input_file: str, output_file: str, password: str):
    """Encrypt a JSONL file."""
    # Generate key
    key, salt = generate_key(password)
    f = Fernet(key)
    
    # Encrypt file
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'wb') as outfile:
        # Save salt
        outfile.write(salt)
        
        # Read entire file content
        content = infile.read()
        lines = content.splitlines()
        
        # Encrypt each line
        for i, line in enumerate(lines):
            if line:  # Skip empty lines
                encrypted_data = f.encrypt(line.encode())
                outfile.write(encrypted_data)
                # Add newline except for the last line
                if i < len(lines) - 1:
                    outfile.write(b'\n')

def decrypt_file(input_file: str, output_file: str, password: str):
    """Decrypt an encrypted file."""
    try:
        with open(input_file, 'rb') as infile:
            # Read salt
            salt = infile.read(16)
            
            # Generate key
            key, _ = generate_key(password, salt)
            f = Fernet(key)
            
            # Decrypt file
            with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
                # Read entire file content
                content = infile.read()
                lines = content.splitlines()
                
                # Decrypt each line
                for i, line in enumerate(lines):
                    if line:  # Skip empty lines
                        try:
                            decrypted_data = f.decrypt(line)
                            outfile.write(decrypted_data.decode())
                            # Add newline except for the last line
                            if i < len(lines) - 1:
                                outfile.write('\n')
                        except InvalidToken:
                            print("Error: Incorrect password.")
                            sys.exit(1)
    except Exception as e:
        print("Error: An error occurred while decrypting the file.")
        print(f"Detailed error: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='JSONL file encryption/decryption tool')
    parser.add_argument('mode', choices=['encrypt', 'decrypt'], help='Encryption or decryption mode')
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('output_file', help='Output file path')
    parser.add_argument('password', help='Password for encryption/decryption')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'encrypt':
            encrypt_file(args.input_file, args.output_file, args.password)
            print(f"File has been encrypted and saved to {args.output_file}")
        else:
            decrypt_file(args.input_file, args.output_file, args.password)
            print(f"File has been decrypted and saved to {args.output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 