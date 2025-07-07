# Security Guide for Smart Health Agent

This guide explains how to securely store credentials and sensitive information for the Smart Health Agent.

## 🔐 Credential Storage Methods

### Method 1: .env File (Recommended for Development)

**Pros:**
- Easy to set up
- Automatically loaded by the application
- Never committed to git
- Works across different environments

**Setup:**
```bash
# Option A: Interactive setup
python setup_garmin.py

# Option B: Manual setup
cp .env.example .env
# Edit .env with your credentials
```

**File structure:**
```
GARMIN_USERNAME=your_email@example.com
GARMIN_PASSWORD=your_password
OLLAMA_HOST=http://localhost:11434
```

### Method 2: Environment Variables (Recommended for Production)

**Pros:**
- Most secure for production environments
- No files with credentials on disk
- Standard practice for containerized applications

**Setup:**
```bash
# Set environment variables
export GARMIN_USERNAME="your_email@example.com"
export GARMIN_PASSWORD="your_password"
export OLLAMA_HOST="http://localhost:11434"

# Or add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
echo 'export GARMIN_USERNAME="your_email@example.com"' >> ~/.bashrc
echo 'export GARMIN_PASSWORD="your_password"' >> ~/.bashrc
```

### Method 3: System Keychain (macOS/Linux)

**Pros:**
- OS-level security
- Encrypted storage
- No plain text files

**Setup (macOS):**
```bash
# Store credentials in keychain
security add-generic-password -a $USER -s garmin-connect -w your_password
security add-generic-password -a $USER -s garmin-username -w your_email@example.com

# Retrieve in Python
import subprocess
def get_keychain_password(service):
    result = subprocess.run(['security', 'find-generic-password', '-s', service, '-w'], 
                          capture_output=True, text=True)
    return result.stdout.strip()
```

### Method 4: Cloud Secret Management (Production)

**Pros:**
- Enterprise-grade security
- Centralized management
- Audit trails

**Examples:**
- AWS Secrets Manager
- Google Cloud Secret Manager
- Azure Key Vault
- HashiCorp Vault

## 🚫 What NOT to Do

### ❌ Never Hardcode Credentials
```python
# DON'T DO THIS
username = "my_email@example.com"
password = "my_password123"
```

### ❌ Never Commit Credentials to Git
```bash
# DON'T DO THIS
git add .env  # This will commit your credentials!
```

### ❌ Never Share Credentials in Logs
```python
# DON'T DO THIS
print(f"Username: {username}")  # Credentials in logs!
logger.info(f"Password: {password}")  # Credentials in logs!
```

## 🔒 Security Best Practices

### 1. Use Strong Passwords
- Minimum 12 characters
- Mix of uppercase, lowercase, numbers, symbols
- Avoid common patterns

### 2. Enable Two-Factor Authentication
- Enable 2FA on your Garmin account
- Use authenticator apps, not SMS

### 3. Regular Credential Rotation
- Change passwords regularly
- Monitor for suspicious activity

### 4. Principle of Least Privilege
- Only grant necessary permissions
- Use dedicated accounts for applications

### 5. Secure Development Practices
- Use `.gitignore` to exclude sensitive files
- Review code before committing
- Use pre-commit hooks to catch secrets

## 🛡️ Environment-Specific Security

### Development
- Use `.env` files with `python-dotenv`
- Keep credentials local only
- Use synthetic data when possible

### Testing
- Use dedicated test accounts
- Never use production credentials
- Use environment-specific configuration

### Production
- Use environment variables or secret management
- Implement proper logging (no credentials)
- Use container security best practices
- Regular security audits

## 🔍 Monitoring and Auditing

### Log Security
```python
# Good: Log actions, not credentials
logger.info("Garmin authentication successful")
logger.info(f"Retrieved data for user: {user_id}")

# Bad: Log credentials
logger.info(f"Username: {username}")  # DON'T DO THIS
```

### Error Handling
```python
# Good: Generic error messages
except GarminConnectError as e:
    logger.error("Authentication failed")
    # Don't expose credential details

# Bad: Exposing sensitive information
except Exception as e:
    logger.error(f"Login failed for {username}: {e}")  # DON'T DO THIS
```

## 📋 Security Checklist

- [ ] Credentials stored securely (not hardcoded)
- [ ] `.env` file in `.gitignore`
- [ ] No credentials in logs
- [ ] Strong passwords used
- [ ] 2FA enabled on accounts
- [ ] Regular credential rotation
- [ ] Environment-specific configuration
- [ ] Security monitoring in place

## 🆘 Security Incident Response

If credentials are compromised:

1. **Immediate Actions:**
   - Change passwords immediately
   - Revoke any tokens/sessions
   - Check for unauthorized access

2. **Investigation:**
   - Review logs for suspicious activity
   - Identify how credentials were exposed
   - Document the incident

3. **Prevention:**
   - Implement additional security measures
   - Update security procedures
   - Train team on security best practices

## 📞 Security Support

For security-related questions or incidents:
- Review this guide first
- Check the project's security policy
- Contact the maintainers through GitHub issues
- Consider professional security consultation for production deployments 