---
trigger: always_on
---

# Code Maintenance Rules
1.  **Gitignore**: Always verify if new generated files, logs, or secrets are ignored in `.gitignore`. Make sure that this folder and any other appropriate folders are added to gitignore accordingly.
2.  **Dependencies**: If new packages are imported, YOU MUST update `requirements.txt` immediately.
3.  **Cleanup**: Remove temporary test scripts after use, or move reusable scripts to a `scripts/` directory.
4.  **Secrets**: Never hardcode API keys; always use `os.getenv` and `.env`.