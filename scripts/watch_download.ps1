while ($true) {
    $count = (Get-ChildItem "data\raw\images" -File -ErrorAction SilentlyContinue | Measure-Object).Count
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts] images downloaded: $count / 600"
    if ($count -ge 600) { break }
    Start-Sleep -Seconds 3
}
