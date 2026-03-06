#!/usr/bin/env bash

set -euo pipefail

# Bootstrap script for PowderKeyGen GNU Radio dependencies.
# It installs system packages, clones required OOT modules, and builds them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${SCRIPT_DIR}/external"
PY_REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
CONFIG_FILE="${SCRIPT_DIR}/config.json"
NPROC="$(nproc)"

run_as_root() {
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

log() {
    printf "\n[%s] %s\n" "$(date +'%H:%M:%S')" "$*"
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1"
        exit 1
    fi
}

clone_or_update_repo() {
    local repo_url="$1"
    local dest_dir="$2"
    local branch="${3:-}"

    if [[ -d "${dest_dir}/.git" ]]; then
        log "Updating ${dest_dir}"
        git -C "${dest_dir}" fetch --all --prune
    else
        log "Cloning ${repo_url} -> ${dest_dir}"
        git clone "${repo_url}" "${dest_dir}"
    fi

    if [[ -n "${branch}" ]]; then
        log "Checking out ${branch} in ${dest_dir}"
        git -C "${dest_dir}" checkout "${branch}"
        git -C "${dest_dir}" pull --ff-only origin "${branch}"
    fi
}

build_and_install_module() {
    local src_dir="$1"

    log "Configuring ${src_dir}"
    cmake -S "${src_dir}" -B "${src_dir}/build"

    log "Building ${src_dir}"
    cmake --build "${src_dir}/build" -j"${NPROC}"

    log "Installing ${src_dir}"
    run_as_root cmake --install "${src_dir}/build"
    run_as_root ldconfig
}

install_python_packages() {
    local pip_args=("$@")

    if python3 -m pip install "${pip_args[@]}"; then
        return 0
    fi

    log "System pip install failed; retrying with --user"
    python3 -m pip install --user "${pip_args[@]}"
}

detect_host_ip() {
    if [[ -n "${POWDERKEYGEN_NODE_IP:-}" ]]; then
        echo "${POWDERKEYGEN_NODE_IP}"
        return 0
    fi

    local route_ip
    route_ip="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')"
    if [[ -n "${route_ip}" ]]; then
        echo "${route_ip}"
        return 0
    fi

    hostname -I 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i !~ /^127\./){print $i; exit}}'
}

update_config_ip() {
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        log "No config file found at ${CONFIG_FILE}; skipping node.ip update"
        return 0
    fi

    local host_ip
    host_ip="$(detect_host_ip)"
    if [[ -z "${host_ip}" ]]; then
        log "Could not determine host IPv4 address; leaving config.json unchanged"
        return 0
    fi

    log "Updating config node.ip to ${host_ip}"
    python3 - "${CONFIG_FILE}" "${host_ip}" <<'PY'
import json
import sys

config_path = sys.argv[1]
host_ip = sys.argv[2]

with open(config_path, "r", encoding="utf-8") as fh:
    config = json.load(fh)

config.setdefault("node", {})
config["node"]["ip"] = host_ip

with open(config_path, "w", encoding="utf-8") as fh:
    json.dump(config, fh, indent=4)
    fh.write("\n")
PY
}

main() {
    require_cmd git

    log "Updating apt package index"
    run_as_root apt-get update

    log "Installing apt dependencies"
    run_as_root apt-get install -y \
        build-essential \
        cmake \
        git \
        gnuradio \
        libboost-all-dev \
        libfftw3-dev \
        libgmp-dev \
        libsndfile1-dev \
        libuhd-dev \
        libvolk2-dev \
        pkg-config \
        pybind11-dev \
        python3-dev \
        python3-pip

    log "Upgrading pip"
    install_python_packages --upgrade pip

    if [[ -f "${PY_REQUIREMENTS_FILE}" ]]; then
        log "Installing Python requirements from ${PY_REQUIREMENTS_FILE}"
        install_python_packages -r "${PY_REQUIREMENTS_FILE}"
    else
        log "No Python requirements file found at ${PY_REQUIREMENTS_FILE}; skipping"
    fi

    mkdir -p "${DEPS_DIR}"

    clone_or_update_repo "https://github.com/bastibl/gr-foo.git" \
        "${DEPS_DIR}/gr-foo" "maint-3.10"
    clone_or_update_repo "https://github.com/joseasanvil/gr-ieee802-11.git" \
        "${DEPS_DIR}/gr-ieee802-11" "maint-3.10"
    clone_or_update_repo "https://github.com/joseasanvil/gr-delta_pulse.git" \
        "${DEPS_DIR}/gr-delta_pulse" "main"

    # Build order matters: gr-ieee802-11 depends on blocks provided by gr-foo.
    build_and_install_module "${DEPS_DIR}/gr-foo"
    build_and_install_module "${DEPS_DIR}/gr-ieee802-11"
    build_and_install_module "${DEPS_DIR}/gr-delta_pulse"
    update_config_ip

    log "Dependency setup complete."
    echo "Run GNU Radio Companion after this: gnuradio-companion"
}

main "$@"
