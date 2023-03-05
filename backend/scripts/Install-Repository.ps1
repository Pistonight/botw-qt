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

function Install-Repository {
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
        $_Params = @{
            ItemType = "Directory"
            Path = ${ProjectDepsRoot}
            ErrorAction = "SilentlyContinue"
        }
        Log-Information "Creating $ProjectDepsRoot"
        New-Item @_Params
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

    # Setup Prebuilt OBS Dependencies
    Install-ObsDependencies

    # Setup OBS Studio Repository
    Install-Obs

    # Setup OpenCV for Windows
    Install-OpenCV

    # Setup TensorFlow
    # Install-TensorFlow

    Log-Information "Configuration Done. Run Build.ps1 to build"

}

Install-Repository
