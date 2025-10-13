#!/bin/bash

# PettingLLMs Documentation Build Script

set -e  # Exit on error

echo "🚀 Building PettingLLMs Documentation..."

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo "❌ mkdocs not found. Installing documentation dependencies..."
    pip install -r docs/requirements.txt
fi

# Parse command line arguments
COMMAND=${1:-build}

case $COMMAND in
    serve)
        echo "📚 Serving documentation with live reload..."
        echo "🌐 Open http://localhost:8000 in your browser"
        mkdocs serve
        ;;
    build)
        echo "🔨 Building static site..."
        mkdocs build
        echo "✅ Documentation built successfully!"
        echo "📁 Output directory: site/"
        ;;
    deploy)
        echo "🚀 Deploying documentation to GitHub Pages..."
        mkdocs gh-deploy
        echo "✅ Documentation deployed successfully!"
        ;;
    clean)
        echo "🧹 Cleaning build artifacts..."
        rm -rf site/
        echo "✅ Cleaned!"
        ;;
    *)
        echo "Usage: $0 {build|serve|deploy|clean}"
        echo ""
        echo "Commands:"
        echo "  build  - Build static documentation (default)"
        echo "  serve  - Serve with live reload"
        echo "  deploy - Deploy to GitHub Pages"
        echo "  clean  - Remove build artifacts"
        exit 1
        ;;
esac

