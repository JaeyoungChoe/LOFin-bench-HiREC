# 🔐 JSONL File Encryption/Decryption Tool

This tool provides encryption and decryption functionality for JSONL files using the Fernet symmetric encryption scheme (`cryptography` library, AES-128-CBC with PKCS7 padding). The goal is to securely distribute evaluation data and **prevent data leakage**.

## 🔧 Requirements

- Python 3.8+
- Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚙️ Usage

### Encrypt
```bash
python encrypt_decrypt.py encrypt input.jsonl output.encrypt "your_password"
```

### Decrypt
```bash
python encrypt_decrypt.py decrypt input.encrypt output.jsonl "your_password"
```

## 🛡️ Notes

- Each line is encrypted independently to support large files.
- Salt-based password-derived keys (PBKDF2 + SHA256) are used.
- The password is **not stored** and must be securely provided.

> ⚠️ **The decryption password is provided separately. Please check the linked Google Drive to obtain it.**

## 📁 Encrypted Dataset Structure

- `all_test.encrypt`: All test set
- `by_answer_type/`: Subsets by answer type
  - `textual_test.encrypt`
  - `numeric_table_test.encrypt`
  - `numeric_text.encrypt`
- `by_data_source/`: Subsets by dataset origin
  - `finqa_test.encrypt`
  - `financebench_test.encrypt`
  - `secqa_test.encrypt`

---