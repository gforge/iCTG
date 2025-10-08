#!/usr/bin/env bash

set -euo pipefail

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

count_down() {
    local seconds=${1:-5}
    local action=${2:-"Continuing"}
    while [ $seconds -gt 0 ]; do
        echo -ne "${CYAN}$action in $seconds seconds...${NC}\r"
        sleep 1
        : $((seconds--))
    done
    echo -ne "\n"
}

# Check if we're already in a tmux session
if [ -z "${TMUX:-}" ]; then
    # Check if tmux is installed
    if command -v tmux &> /dev/null; then
        # Check for existing ictg-convert sessions
        EXISTING_SESSIONS=$(tmux list-sessions 2>/dev/null | grep -E "^ictg-convert-" | cut -d: -f1 || true)

        if [ -n "$EXISTING_SESSIONS" ]; then
            echo -e "${GREEN}${BOLD}Found existing conversion session(s):${NC}"
            echo "$EXISTING_SESSIONS" | nl
            echo ""
            echo -ne "${YELLOW}Attach to an existing session? ${BOLD}(Y/n):${NC} "
            read -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
                # Count sessions
                SESSION_COUNT=$(echo "$EXISTING_SESSIONS" | wc -l)
                if [ "$SESSION_COUNT" -eq 1 ]; then
                    SESSION_TO_ATTACH="$EXISTING_SESSIONS"
                else
                    echo -e "${CYAN}Enter session number to attach (1-$SESSION_COUNT):${NC}"
                    read -r SESSION_NUM
                    SESSION_TO_ATTACH=$(echo "$EXISTING_SESSIONS" | sed -n "${SESSION_NUM}p")
                fi
                echo -e "${GREEN}Attaching to session: ${BOLD}$SESSION_TO_ATTACH${NC}"
                count_down 3 "Attaching"
                tmux attach -t "$SESSION_TO_ATTACH"
                exit 0
            fi
        fi

        echo -e "${YELLOW}${BOLD}Long-running conversion detected.${NC} You are not in a tmux session."
        echo -e "${CYAN}It's recommended to run this in tmux to avoid interruption.${NC}"
        echo -ne "${YELLOW}Start a new tmux session? ${BOLD}(Y/n):${NC} "
        read -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            # Create a session name with timestamp
            SESSION_NAME="ictg-convert-$(date +%Y%m%d-%H%M%S)"
            echo -e "${GREEN}Starting tmux session: ${BOLD}$SESSION_NAME${NC}"
            echo -e "${CYAN}The conversion will run inside tmux. You can:${NC}"
            echo -e "  ${BLUE}•${NC} Detach: ${BOLD}Ctrl+B${NC} then ${BOLD}D${NC} (conversion continues in background)"
            echo -e "  ${BLUE}•${NC} Reattach later: ${BOLD}tmux attach -t $SESSION_NAME${NC}"
            echo ""
            count_down 5 "Starting tmux"
            tmux new-session -s "$SESSION_NAME" "$0" "$@"
            exit 0
        fi
    else
        echo -e "${RED}${BOLD}Warning:${NC} tmux is not installed. For long-running conversions, consider installing tmux."
        echo -e "${CYAN}Install with:${NC} ${BOLD}sudo apt-get install tmux${NC} (Debian/Ubuntu) or ${BOLD}brew install tmux${NC} (macOS)"
        echo -ne "${YELLOW}Continue without tmux? ${BOLD}(y/N):${NC} "
        read -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Execute the main Python script using uv
uv run python -m src.ictg.convert.main "$@"
