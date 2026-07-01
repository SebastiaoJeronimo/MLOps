# Install kind (if missing) and add it to PATH for this PowerShell session.
# Usage (from serving/):  . .\k8s\setup-kind.ps1

function Find-KindExe {
    $onPath = Get-Command kind -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    $wingetRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path $wingetRoot) {
        $found = Get-ChildItem (Join-Path $wingetRoot "Kubernetes.kind_*") -Recurse -Filter kind.exe -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    return $null
}

$kindExe = Find-KindExe

if (-not $kindExe) {
    Write-Host "kind not found - installing via winget..."
    winget install --id Kubernetes.kind -e --accept-source-agreements --accept-package-agreements
    $kindExe = Find-KindExe
}

if (-not $kindExe) {
    throw "kind is still not available. Install manually: winget install Kubernetes.kind"
}

$kindDir = Split-Path $kindExe -Parent
if ($env:Path -notlike "*$kindDir*") {
    $env:Path = $kindDir + ";" + $env:Path
}

Write-Host "kind:" $kindExe
kind version

if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    Write-Host "kubectl not on PATH - install Docker Desktop or: winget install Kubernetes.kubectl"
} else {
    kubectl version --client
}
