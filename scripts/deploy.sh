#!/bin/bash
# ============================================
# Script de Déploiement Bug Predictor
# ============================================

set -e  # Arrêter en cas d'erreur

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="bug-predictor"
DOCKER_IMAGE="$APP_NAME:latest"
CONTAINER_NAME="$APP_NAME-app"
PORT=8501

# Fonctions
print_header() {
    echo -e "${BLUE}========================================"
    echo -e "$1"
    echo -e "========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# ============================================
# STAGE 1: LINT
# ============================================
stage_lint() {
    print_header "STAGE 1: LINT & CODE QUALITY"

    print_info "Running Flake8..."
    flake8 src/ tests/ --count --statistics || print_warning "Flake8 warnings detected"

    print_info "Running Black..."
    black --check src/ tests/ || print_warning "Formatting issues detected"

    print_info "Running isort..."
    isort --check-only src/ tests/ || print_warning "Import order issues detected"

    print_success "Lint stage completed"
}

# ============================================
# STAGE 2: TEST
# ============================================
stage_test() {
    print_header "STAGE 2: TESTS"

    print_info "Running unit tests..."
    pytest tests/ -m "unit" -v --tb=short --cov=src --cov-report=term-missing

    print_info "Running integration tests..."
    pytest tests/ -m "integration" -v --tb=short

    print_success "Test stage completed"
}

# ============================================
# STAGE 3: BUILD
# ============================================
stage_build() {
    print_header "STAGE 3: BUILD DOCKER IMAGE"

    print_info "Building Docker image..."
    docker build -t $DOCKER_IMAGE .

    print_info "Testing Docker image..."
    docker run --rm $DOCKER_IMAGE --version || print_info "Version check skipped"

    print_success "Build stage completed"
    print_info "Image: $DOCKER_IMAGE"
}

# ============================================
# STAGE 4: DEPLOY
# ============================================
stage_deploy() {
    print_header "STAGE 4: DEPLOY"

    # Arrêter conteneur existant
    print_info "Stopping existing container..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true

    # Démarrer nouveau conteneur
    print_info "Starting new container..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:8501 \
        -v $(pwd)/models:/app/models:ro \
        -v $(pwd)/data:/app/data:ro \
        -v $(pwd)/results:/app/results \
        --restart unless-stopped \
        $DOCKER_IMAGE

    # Attendre démarrage
    print_info "Waiting for application to start..."
    sleep 10

    # Health check
    print_info "Performing health check..."
    if curl -f http://localhost:$PORT/_stcore/health > /dev/null 2>&1; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
        docker logs $CONTAINER_NAME
        exit 1
    fi

    print_success "Deploy stage completed"
    print_info "Application running at: http://localhost:$PORT"
}

# ============================================
# COMMANDES
# ============================================
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  lint       Run linting only"
    echo "  test       Run tests only"
    echo "  build      Build Docker image only"
    echo "  deploy     Deploy locally"
    echo "  full       Run full pipeline (lint → test → build → deploy)"
    echo "  stop       Stop running container"
    echo "  logs       Show container logs"
    echo "  status     Show container status"
    echo "  help       Show this help message"
    echo ""
}

stop_container() {
    print_header "STOPPING CONTAINER"
    docker stop $CONTAINER_NAME 2>/dev/null || print_warning "Container not running"
    docker rm $CONTAINER_NAME 2>/dev/null || true
    print_success "Container stopped"
}

show_logs() {
    print_header "CONTAINER LOGS"
    docker logs -f $CONTAINER_NAME
}

show_status() {
    print_header "CONTAINER STATUS"
    docker ps -a | grep $CONTAINER_NAME || print_warning "Container not found"
    echo ""
    print_info "Application URL: http://localhost:$PORT"
}

# ============================================
# MAIN
# ============================================
main() {
    case "${1:-help}" in
        lint)
            stage_lint
            ;;
        test)
            stage_test
            ;;
        build)
            stage_build
            ;;
        deploy)
            stage_deploy
            ;;
        full)
            print_header "🚀 FULL PIPELINE"
            stage_lint
            echo ""
            stage_test
            echo ""
            stage_build
            echo ""
            stage_deploy
            echo ""
            print_header "🎉 PIPELINE COMPLETED SUCCESSFULLY"
            print_info "Application: http://localhost:$PORT"
            ;;
        stop)
            stop_container
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            ;;
        help)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"