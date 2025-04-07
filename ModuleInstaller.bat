@echo off
echo Installing required Python modules...

(
    pip install pandas || echo Failed to install pandas
    pip install networkx || echo Failed to install networkx
    pip install matplotlib || echo Failed to install matplotlib
) || (
    echo An error occurred during installation. please flag the issue on the Github repository
)

echo Installation complete.
pause