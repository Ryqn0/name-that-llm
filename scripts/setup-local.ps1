<#
.SYNOPSIS
    Make Python, uv and the Kaggle CLI work on a Windows PC running Avast HTTPS scanning.

.DESCRIPTION
    WHY THIS SCRIPT EXISTS
    ----------------------
    Avast's "Web Shield" sits in the middle of every encrypted (HTTPS) connection:
    it decrypts the traffic to scan it, then re-encrypts it with its OWN certificate.
    Browsers trust that certificate because Avast adds it to the Windows certificate
    store. Developer tools often don't, and Avast also injects two environment
    variables into every program it starts. On this machine that caused 3 distinct
    failures (diagnosed 2026-10-23):

      SYMPTOM                                   CAUSE                          FIX (below)
      ----------------------------------------  -----------------------------  -----------
      Python crashes:                           Avast sets SSLKEYLOGFILE to a  Step 6
        "OPENSSL_Uplink ... no OPENSSL_Applink"  named pipe; Python's ssl
                                                 module tries to open it.
      uv: "invalid peer certificate:            uv ships its own list of       Step 5
        UnknownIssuer"                           trusted certificates, without
                                                 Avast's.
      Python requests:                          Python uses certifi's list,    Steps 3-5
        "CERTIFICATE_VERIFY_FAILED"              which lacks Avast's cert.

    Why a COMBINED bundle (certifi + Avast) and not Avast's certificate alone?
    Avast doesn't intercept every site. Kaggle's API goes through Avast, but data
    downloads from storage.googleapis.com come straight from Google with Google's
    real certificate. A bundle with only Avast's cert would break those downloads,
    so we trust both.

    The alternative is to turn off "HTTPS scanning" in Avast's Web Shield settings,
    which makes this script unnecessary. That's a security choice for you to make,
    so this script works around the scanning instead of disabling it.

    SAFE TO RE-RUN: every step checks the current state first, so a second run
    changes nothing. Re-run it whenever Avast updates its certificate.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup-local.ps1
    pwsh       -ExecutionPolicy Bypass -File scripts\setup-local.ps1
    Run it once in EACH PowerShell version you use. Windows PowerShell 5.1 and
    PowerShell 7 keep separate profiles (see Step 6).
#>

# Stop at the first error instead of continuing in a half-configured state.
$ErrorActionPreference = 'Stop'

# --- Step 1: is Avast here at all? -------------------------------------------
# Avast keeps its interception certificate here. No file means no Avast HTTPS
# scanning, so there's nothing to fix. Exit cleanly, so the script is harmless on
# any other machine.
$avastCert = 'C:\ProgramData\Avast Software\Avast\wscert.pem'
if (-not (Test-Path $avastCert)) {
    Write-Host 'No Avast certificate found, nothing to do.'
    exit 0
}
Write-Host "[found] Avast certificate: $avastCert"

# --- Step 2: fix THIS process first ------------------------------------------
# The next steps run uv and Python, which would crash or fail on TLS just like
# everything else. These two lines affect only this script's own process; the
# permanent fixes come in Steps 5 and 6.
Remove-Item Env:SSLKEYLOGFILE -ErrorAction SilentlyContinue  # prevents the OPENSSL_Uplink crash
$env:UV_SYSTEM_CERTS = '1'                                   # uv: trust the Windows certificate store

# --- Step 3: find certifi's certificate bundle -------------------------------
# certifi is the standard list of public root certificates most Python tools use.
# uv fetches it into a throwaway environment (--no-project means "don't touch any
# project"). -I = isolated mode: Python ignores environment variables and the
# current folder, so nothing local can interfere.
$certifiBundle = (& uv run --no-project --with certifi python -I -c "import certifi; print(certifi.where())").Trim()
if (-not (Test-Path $certifiBundle)) { throw "certifi bundle not found at '$certifiBundle'" }
Write-Host "[found] certifi bundle: $certifiBundle"

# --- Step 4: write the combined bundle ---------------------------------------
# A PEM bundle is just certificates pasted one after another, so concatenating
# the two files is enough. We write UTF-8 *without* BOM through .NET directly:
# Windows PowerShell 5.1's Set-Content -Encoding UTF8 would add a BOM, and some
# tools choke on that.
$certDir = Join-Path $HOME '.certs'
$bundle  = Join-Path $certDir 'combined-ca.pem'
New-Item -ItemType Directory -Force -Path $certDir | Out-Null    # -Force: no error if it exists
$combined = [System.IO.File]::ReadAllText($certifiBundle).TrimEnd() + "`n" +
            [System.IO.File]::ReadAllText($avastCert).TrimEnd() + "`n"
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
if ((Test-Path $bundle) -and ([System.IO.File]::ReadAllText($bundle) -eq $combined)) {
    Write-Host "[skip]  bundle already up to date: $bundle"
} else {
    [System.IO.File]::WriteAllText($bundle, $combined, $utf8NoBom)
    $count = ([regex]::Matches($combined, 'BEGIN CERTIFICATE')).Count
    Write-Host "[write] bundle with $count certificates: $bundle"
}

# --- Step 5: point the tools at it, permanently (user-level env vars) --------
# "User" scope persists across reboots and applies to every NEW terminal or
# program, but not to windows that are already open (hence the reminder in Step 8).
#   REQUESTS_CA_BUNDLE : read by the 'requests' library (Kaggle CLI, wandb, ...)
#   SSL_CERT_FILE      : read by Python's own ssl module and OpenSSL-based tools
#   UV_SYSTEM_CERTS    : tells uv to trust the Windows store, which includes Avast
$wanted = [ordered]@{
    REQUESTS_CA_BUNDLE = $bundle
    SSL_CERT_FILE      = $bundle
    UV_SYSTEM_CERTS    = '1'
}
foreach ($name in $wanted.Keys) {
    $current = [Environment]::GetEnvironmentVariable($name, 'User')
    if ($current -eq $wanted[$name]) {
        Write-Host "[skip]  $name already set"
    } else {
        [Environment]::SetEnvironmentVariable($name, $wanted[$name], 'User')
        Write-Host "[set]   $name = $($wanted[$name])"
    }
}

# --- Step 6: remove SSLKEYLOGFILE in every new PowerShell session ------------
# Avast injects SSLKEYLOGFILE into each program as it starts, so a permanent
# env-var change can't remove it. But programs inherit their environment from
# the shell that launches them: removing the variable once, at shell startup,
# protects everything run from that terminal. The PowerShell *profile* is the
# script PowerShell runs at startup. 5.1 and 7 each have their own.
$fixLine = 'Remove-Item Env:SSLKEYLOGFILE -ErrorAction SilentlyContinue  # Avast injects this; it crashes Python''s ssl module'
if (-not (Test-Path $PROFILE)) {
    # -Force creates missing parent folders. The file doesn't exist yet, so nothing gets overwritten.
    New-Item -ItemType File -Force -Path $PROFILE | Out-Null
    Write-Host "[new]   created profile: $PROFILE"
}
if (Select-String -Path $PROFILE -SimpleMatch 'Remove-Item Env:SSLKEYLOGFILE' -Quiet) {
    Write-Host "[skip]  profile already removes SSLKEYLOGFILE: $PROFILE"
} else {
    Add-Content -Path $PROFILE -Value $fixLine
    Write-Host "[add]   SSLKEYLOGFILE fix appended to: $PROFILE"
}

# --- Step 7: install the Kaggle CLI as a global tool -------------------------
# 'uv tool install' gives the CLI its own isolated environment and puts kaggle.exe
# on PATH, next to uv itself. If a matching version is already installed, uv does
# nothing. Version 2.2.3+ is required for the new KGAT access tokens and for the
# forum commands.
& uv tool install "kaggle>=2.2.3"
if ($LASTEXITCODE -ne 0) { throw "uv tool install kaggle failed (exit $LASTEXITCODE)" }
Write-Host '[ok]    kaggle CLI installed'

# --- Step 8: done ------------------------------------------------------------
Write-Host ''
Write-Host 'Open a NEW terminal. If Avast updates its certificate, re-run this script.'
