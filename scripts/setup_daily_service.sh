#!/bin/bash

# Setup script for daily RSS analyzer service

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Setting up daily RSS analyzer service..."
echo "Project directory: $PROJECT_DIR"

# Option 1: Cron job setup
setup_cron() {
    echo "Setting up cron job to run once daily at 2 AM..."

    # Create cron job entry
    CRON_JOB="0 2 * * * cd $PROJECT_DIR && ./run_daily.sh >> logs/cron.log 2>&1"

    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

    echo "Cron job added successfully!"
    echo "The RSS analyzer will now run once daily at 2 AM."
    echo "Logs will be written to:"
    echo "  - $PROJECT_DIR/logs/daily_runs.log"
    echo "  - $PROJECT_DIR/logs/cron.log"

    # Show current crontab
    echo ""
    echo "Current crontab:"
    crontab -l
}

# Option 2: macOS launchd service (better for macOS)
setup_launchd() {
    SERVICE_NAME="com.rss-analyzer.daily"
    PLIST_FILE="$HOME/Library/LaunchAgents/$SERVICE_NAME.plist"

    echo "Setting up macOS LaunchAgent..."

    # Create LaunchAgent plist
    cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$SERVICE_NAME</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PROJECT_DIR/run_daily.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/logs/launchd_error.log</string>
</dict>
</plist>
EOF

    # Load the service
    launchctl load "$PLIST_FILE"

    echo "LaunchAgent created and loaded successfully!"
    echo "Service file: $PLIST_FILE"
    echo "The RSS analyzer will run once daily at 2 AM."
    echo "Logs will be written to:"
    echo "  - $PROJECT_DIR/logs/daily_runs.log"
    echo "  - $PROJECT_DIR/logs/launchd.log"
    echo "  - $PROJECT_DIR/logs/launchd_error.log"

    echo ""
    echo "To control the service:"
    echo "  Stop:  launchctl unload $PLIST_FILE"
    echo "  Start: launchctl load $PLIST_FILE"
    echo "  Check: launchctl list | grep $SERVICE_NAME"
}

# Detect OS and choose appropriate method
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS. Choose setup method:"
    echo "1) LaunchAgent (recommended for macOS)"
    echo "2) Cron job"
    read -p "Enter choice (1 or 2): " choice

    case $choice in
        1)
            setup_launchd
            ;;
        2)
            setup_cron
            ;;
        *)
            echo "Invalid choice. Using LaunchAgent..."
            setup_launchd
            ;;
    esac
else
    echo "Detected Linux/Unix. Setting up cron job..."
    setup_cron
fi

echo ""
echo "Setup complete! The RSS analyzer will now run automatically once daily at 2 AM."
echo "You can test the setup by running: ./run_daily.sh"
