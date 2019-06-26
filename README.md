# sandia-blusky

# Getting Started

In this section we provide detailed instructions on how to get up and running
with the Sandia Blusky library.

# Configuring Git


If you're working on Windows, make sure that your Git installation is
configured to handle text-file line-endings correctly.
We recommend using the setting:

core.autocrlf=true

This ensures that Unix-style line-endings are converted to Windows when code is checked out from GitHub, so that you can safely edit those files using Windows editors and conventions. The line-endings are automatically converted back to Unix-style line-endings when you commit. To set this configuration up, use:

git config --global core.autocrlf true

Before making any commits, ensure that the user.name and user.email are configured, so that any changes are properly attributed to the user making them. To set these variables up, use:

git config --global user.name "My Name"
git config --global user.email "my_email@email.com"

To check your current configuration settings, use:

git config --list
