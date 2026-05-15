@echo off
setlocal enabledelayedexpansion

set "target=.\img"
if not exist "%target%" mkdir "%target%"

for %%f in (*.json) do (
    set "basename=%%~nf"
    set "moved="
    for %%e in (jpg jpeg png tiff tif) do (
        if exist "!basename!.%%e" (
            if not exist "%target%\!basename!.%%e" (
                echo Moving !basename!.%%e ...
                move "!basename!.%%e" "%target%\" >nul
                if !errorlevel! equ 0 set "moved=1"
            ) else (
                echo Already exists: %target%\!basename!.%%e
            )
        )
    )
    if defined moved (
        if not exist "%target%\%%f" (
            echo Moving %%f ...
            move "%%f" "%target%\" >nul
        ) else (
            echo Json already exists in target, skipping move of %%f.
        )
    )
)

echo.
echo Done. Check "%target%" folder.
pause
