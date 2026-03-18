#!/bin/bash
# Setup script for WebShop environment
# Run this from the project root: ./scripts/setup_webshop.sh
#
# Prerequisites:
#   - Python 3.8+ (conda recommended)
#   - Java (for Lucene search index)
#   - Git

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEBSHOP_DIR="$PROJECT_ROOT/webshop"

echo "=== WebShop Environment Setup ==="
echo "Project root: $PROJECT_ROOT"
echo "WebShop dir:  $WEBSHOP_DIR"

# 1. Clone WebShop if not present
if [ ! -d "$WEBSHOP_DIR" ]; then
    echo ""
    echo ">>> Cloning WebShop repository..."
    git clone https://github.com/princeton-nlp/webshop.git "$WEBSHOP_DIR"
else
    echo ""
    echo ">>> WebShop already cloned at $WEBSHOP_DIR"
fi

# 2. Create conda env if not exists
if ! conda env list | grep -q "webshop"; then
    echo ""
    echo ">>> Creating conda environment 'webshop' with Python 3.8.13..."
    conda create -n webshop python=3.8.13 -y
else
    echo ""
    echo ">>> Conda environment 'webshop' already exists"
fi

# 3. Run WebShop's own setup
echo ""
echo ">>> Running WebShop setup script..."
echo "    This downloads product data and builds the search index."
echo "    Use -d small for quick testing or -d all for full dataset."
echo ""
echo "    To run manually:"
echo "      conda activate webshop"
echo "      cd $WEBSHOP_DIR"
echo "      ./setup.sh -d all"
echo ""

read -p "Run setup now with full dataset? [y/N]: " run_setup
if [[ "$run_setup" =~ ^[Yy]$ ]]; then
    (
        eval "$(conda shell.bash hook)"
        conda activate webshop
        cd "$WEBSHOP_DIR"
        chmod +x setup.sh
        ./setup.sh -d all
    )
fi

# 4. Set environment variable
echo ""
echo ">>> Setup complete!"
echo ""
echo "Add the following to your .env file:"
echo "  WEBSHOP_PATH=$WEBSHOP_DIR"
echo ""
echo "Or export it:"
echo "  export WEBSHOP_PATH=$WEBSHOP_DIR"
echo ""
echo "To verify the installation:"
echo "  conda activate webshop"
echo "  python -c \"import sys; sys.path.insert(0, '$WEBSHOP_DIR'); from web_agent_site.envs import WebAgentTextEnv; print('WebShop OK')\""
