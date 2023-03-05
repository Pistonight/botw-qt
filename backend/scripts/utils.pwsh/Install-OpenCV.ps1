function Install-OpenCV {
    if ( ! ( Test-Path function:Log-Output ) ) {
        . $PSScriptRoot/Logger.ps1
    }

    if ( ! ( Test-Path function:Ensure-Location ) ) {
        . $PSScriptRoot/Ensure-Location.ps1
    }

    if ( ! ( Test-Path function:Invoke-GitCheckout ) ) {
        . $PSScriptRoot/Invoke-GitCheckout.ps1
    }

    if ( ! ( Test-Path function:Invoke-External ) ) {
        . $PSScriptRoot/Invoke-External.ps1
    }

    Log-Information 'Setting up OpenCV...'

    $OpenCVVersion = $BuildSpec.dependencies.'opencv'.version
    $OpenCVBaseUrl = $BuildSpec.dependencies.'opencv'.baseUrl
    $OpenCVLabel = $BuildSpec.dependencies.'opencv'.label
    $OpenCVHash = $BuildSpec.dependencies.'opencv'.hashes."windows"

    if ( $OpenCVVersion -eq '' ) {
        throw 'No opencv version found in buildspec.json.'
    }

    Push-Location -Stack BuildTemp
    Ensure-Location -Path "$(Resolve-Path -Path "${ProjectDepsRoot}")"

    $Filename = "opencv-${OpenCVVersion}-windows.exe"
    $Uri = "${OpenCVBaseUrl}/${OpenCVVersion}/${Filename}"

    if ( ! ( Test-Path -Path $Filename ) ) {
        $Params = @{
            UserAgent = 'NativeHost'
            Uri = $Uri
            OutFile = $Filename
            UseBasicParsing = $true
            ErrorAction = 'Stop'
        }

        Invoke-WebRequest @Params
        Log-Status "Downloaded ${OpenCVLabel}."
    } else {
        Log-Status "Found downloaded ${OpenCVLabel}."
    }

    $_FileHash = Get-FileHash -Path $Filename -Algorithm SHA256

    if ( $_FileHash.Hash.ToLower() -ne $OpenCVHash ) {
        throw "Checksum of downloaded ${OpenCVLabel} does not match specification. Expected '${_Hash}', 'found $(${_FileHash}.Hash.ToLower())'"
    }
    Log-Status "Checksum of downloaded ${OpenCVLabel} matches."

    Expand-ArchiveExt -Path ${Filename} -DestinationPath . -Force
    
    Pop-Location -Stack BuildTemp
}
