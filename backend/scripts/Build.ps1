[CmdletBinding()]
param(
    [ValidateSet('Debug', 'RelWithDebInfo', 'Release', 'MinSizeRel')]
    [string] $Configuration = 'RelWithDebInfo',
    [ValidateSet('x86', 'x64')]
    [string] $Target,
    [ValidateSet('Visual Studio 17 2022', 'Visual Studio 16 2019')]
    [string] $CMakeGenerator
)

$ErrorActionPreference = 'Stop'

if ( $DebugPreference -eq 'Continue' ) {
    $VerbosePreference = 'Continue'
    $InformationPreference = 'Continue'
}

if ( $PSVersionTable.PSVersion -lt '7.0.0' ) {
    Write-Warning 'The obs-deps PowerShell build script requires PowerShell Core 7. Install or upgrade your PowerShell version: https://aka.ms/pscore6'
    exit 2
}

function Build {
    trap {
        Pop-Location -Stack BuildTemp -ErrorAction 'SilentlyContinue'
        Write-Error $_
        exit 2
    }

    $ScriptHome = $PSScriptRoot
    $ProjectRoot = Resolve-Path -Path "$PSScriptRoot/.."
    $BuildSpecFile = "${ProjectRoot}/buildspec.json"

    $UtilityFunctions = Get-ChildItem -Path $PSScriptRoot/utils.pwsh/*.ps1 -Recurse

    foreach($Utility in $UtilityFunctions) {
        Write-Debug "Loading $($Utility.FullName)"
        . $Utility.FullName
    }

    $ProjectDepsRoot = "${ProjectRoot}/deps"
    if ( ! ( Test-Path $ProjectDepsRoot ) ) {
        throw "Cannot find deps folder. Run Install-Repository first."
    }

    $BuildSpec = Get-Content -Path ${BuildSpecFile} -Raw | ConvertFrom-Json
    $ProductName = $BuildSpec.name
    $ProductVersion = $BuildSpec.version

    if ( $script:Target -eq '' ) { $script:Target = $script:HostArchitecture }

    $script:DepsVersion = $BuildSpec.dependencies."prebuilt".version
    $script:QtVersion = $BuildSpec.platformConfig."windows-${script:Target}".qtVersion
    $script:PlatformSDK = $BuildSpec.platformConfig."windows-${script:Target}".platformSDK

    if ( $CmakeGenerator -eq '' ) {
        $CmakeGenerator = $BuildSpec.platformConfig."windows-${script:Target}".visualStudio
    }

    (Get-Content -Path ${ProjectRoot}/CMakeLists.txt -Raw) `
        -replace "project\((.*) VERSION (.*)\)", "project(${ProductName} VERSION ${ProductVersion})" `
        | Out-File -Path ${ProjectRoot}/CMakeLists.txt -NoNewline

    Push-Location -Stack BuildTemp
    Ensure-Location $ProjectRoot
    $DepsPath = "plugin-deps-${script:DepsVersion}-qt${script:QtVersion}-${script:Target}"
    $CmakeArgs = @(
        '-G', "$CmakeGenerator"
        "-DCMAKE_SYSTEM_VERSION=${script:PlatformSDK}"
        "-DCMAKE_GENERATOR_PLATFORM=$(if (${script:Target} -eq "x86") { "Win32" } else { "x64" })"
        "-DCMAKE_BUILD_TYPE=${Configuration}"
        "-DCMAKE_PREFIX_PATH:PATH=$(Resolve-Path -Path "${ProjectDepsRoot}/obs-build-dependencies/${DepsPath}")"
        "-DQT_VERSION=${script:QtVersion}"
        "-DOpenCV_DIR=$(Resolve-Path -Path "${ProjectDepsRoot}/opencv/build/x64/vc16/lib")"
        # "-DTesseract_DIR=$(Resolve-Path -Path "${ProjectDepsRoot}/vcpkg/packages/tesseract_x64-windows/share/tesseract")"
        # "-DLeptonica_DIR=$(Resolve-Path -Path "${ProjectDepsRoot}/vcpkg/packages/leptonica_x64-windows/share/leptonica")"
        # "-DLibArchive_LIBRARY=$(Resolve-Path -Path "${ProjectDepsRoot}/vcpkg/packages/libarchive_x64-windows/lib/archive.lib")"
        # "-DLibArchive_INCLUDE_DIR=$(Resolve-Path -Path "${ProjectDepsRoot}/vcpkg/packages/libarchive_x64-windows/include")"
        "-DObsWebSocket_DIR=$(Resolve-Path -Path "${ProjectDepsRoot}/obs-studio/plugins/obs-websocket")"
    )   

    Log-Debug "Attempting to configure with CMake arguments: $($CmakeArgs | Out-String)"
    Log-Information "Configuring ${ProductName}..."
    Invoke-External cmake -S . -B build_${script:Target} @CmakeArgs

    $CmakeArgs = @(
        '--config', "${Configuration}"
    )

    if ( $VerbosePreference -eq 'Continue' ) {
        $CmakeArgs+=('--verbose')
    }

    Log-Information "Building ${ProductName}..."
    Invoke-External cmake --build "build_${script:Target}" @CmakeArgs
    Log-Information "Install ${ProductName}..."
    Invoke-External cmake --install "build_${script:Target}" --prefix "${ProjectRoot}/release" @CmakeArgs

    Pop-Location -Stack BuildTemp

    # Copy the dependent dlls to the release folder
    Copy-Item -Path "${ProjectDepsRoot}/opencv/build/x64/vc16/bin/opencv_world470.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"

    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/archive.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/zlib1.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/bz2.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/liblzma.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/lz4.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/zstd.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/libcrypto-3-x64.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/leptonica-1.83.1.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/gif.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/jpeg62.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/openjp2.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/libpng16.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/tiff.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/libwebp.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/libsharpyuv.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/libwebpmux.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"
    # Copy-Item -Path "${ProjectDepsRoot}/vcpkg/installed/x64-windows/bin/tesseract53.dll" -Destination "${ProjectRoot}/release/obs-plugins/64bit"

}

Build
