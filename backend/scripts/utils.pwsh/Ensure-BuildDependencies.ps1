function Ensure-BuildDependencies {
    <#
        .SYNOPSIS
            Check if required build dependencies are installed
        .DESCRIPTION
            Additional tools are required to build the project. This function checks if they are installed.
        .EXAMPLE
            Ensure
    #>

    if ( ! ( Test-Path function:Log-Warning ) ) {
        . $PSScriptRoot/Logger.ps1
    }

    $Tools = "git", "7z", "cmake"
    foreach ( $Binary in $Tools ) {
        Log-Debug "Checking for command $Binary"
        $Found = Get-Command -ErrorAction SilentlyContinue $Binary

        if ( ! $Found ) {
            throw "Command $Binary not found. Please install $Binary to continue."        
        } 
        Log-Status "Found dependency $Binary as $($Found.Source)"
    }
}