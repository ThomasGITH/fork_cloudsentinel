#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly NAMESPACE="cloudsentinel"
readonly ADMIN_POD="plugin-admin"
readonly ADMIN_MANIFEST="${REPOSITORY_ROOT}/k8s/plugin-admin-pod.yml"
KEEP_ADMIN_POD=false
REMOTE_SOURCE=""
ADMIN_POD_TOUCHED=false

usage() {
  cat <<'EOF'
Usage:
  external_plugin.sh publish ABSOLUTE_PLUGIN_DIR --id DETECTOR_ID --version VERSION [--activate] [--keep-admin-pod]
  external_plugin.sh rollback DETECTOR_ID VERSION_OR_SHA256 [--keep-admin-pod]
  external_plugin.sh list [--keep-admin-pod]

EOF
}

die() {
  printf 'External plugin operation failed: %s\n' "$1" >&2
  exit 1
}

cleanup() {
  local exit_code=$?
  set +e
  if [[ -n "${REMOTE_SOURCE}" ]] && command -v kubectl >/dev/null 2>&1; then
    kubectl exec -n "${NAMESPACE}" "${ADMIN_POD}" -- \
      rm -rf "${REMOTE_SOURCE}" >/dev/null 2>&1
  fi
  if [[ "${ADMIN_POD_TOUCHED}" == true && "${KEEP_ADMIN_POD}" != true ]] \
      && command -v kubectl >/dev/null 2>&1; then
    kubectl delete pod -n "${NAMESPACE}" "${ADMIN_POD}" \
      --ignore-not-found=true --wait=false >/dev/null 2>&1
  fi
  if (( exit_code != 0 )); then
    printf 'The plugin was not published. Any previously active release remains unchanged.\n' >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command '$1' is unavailable"
}

validate_detector_id() {
  [[ "$1" =~ ^[a-z][a-z0-9]*([-_][a-z0-9]+)*$ ]] \
    || die "detector ID must start with a lowercase letter and use lowercase letters, digits, hyphens or underscores"
}

validate_selector() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] \
    || die "version or release selector contains unsafe characters"
}

validate_version() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] \
    || die "version contains unsafe characters"
}

validate_local_source() {
  local source_dir=$1
  [[ "${source_dir}" == /* ]] || die "plugin directory must be an absolute path"
  [[ -d "${source_dir}" ]] || die "plugin directory does not exist"
  [[ ! -L "${source_dir}" ]] || die "plugin directory may not be a symbolic link"
  [[ -f "${source_dir}/manifest.yaml" ]] || die "plugin directory is missing manifest.yaml"
  [[ -f "${source_dir}/adapter.py" ]] || die "plugin directory is missing adapter.py"
  [[ -f "${source_dir}/requirements.lock" ]] || die "plugin directory is missing requirements.lock"
}

json_field() {
  local field=$1
  python3 -c \
    'import json, sys; value=json.load(sys.stdin); result=value.get(sys.argv[1]); sys.stdout.write("" if result is None else str(result))' \
    "${field}"
}

prepare_admin_pod() {
  require_command kubectl
  [[ -f "${ADMIN_MANIFEST}" ]] || die "plugin admin manifest is unavailable"

  if kubectl get pod -n "${NAMESPACE}" "${ADMIN_POD}" >/dev/null 2>&1; then
    local existing_label
    existing_label="$(kubectl get pod -n "${NAMESPACE}" "${ADMIN_POD}" \
      -o jsonpath='{.metadata.labels.app}')"
    [[ "${existing_label}" == "plugin-admin" ]] \
      || die "an unrelated pod already uses the plugin administration pod name"
    ADMIN_POD_TOUCHED=true
    if kubectl wait -n "${NAMESPACE}" --for=condition=Ready \
        "pod/${ADMIN_POD}" --timeout=5s >/dev/null 2>&1; then
      printf 'Reusing ready plugin administration pod.\n'
      return
    fi
    printf 'Replacing stale plugin administration pod...\n'
    kubectl delete pod -n "${NAMESPACE}" "${ADMIN_POD}" --wait=true >/dev/null
  fi

  printf 'Creating temporary plugin administration pod...\n'
  kubectl apply -f "${ADMIN_MANIFEST}" >/dev/null
  ADMIN_POD_TOUCHED=true
  if ! kubectl wait -n "${NAMESPACE}" --for=condition=Ready \
      "pod/${ADMIN_POD}" --timeout=120s >/dev/null; then
    die "plugin administration pod did not become Ready within 120 seconds"
  fi
}

publish_plugin() {
  local source_dir=""
  local detector_id=""
  local detector_version=""
  local activate=false

  [[ $# -ge 1 ]] || die "publish requires an absolute plugin directory"
  source_dir=$1
  shift
  while (($#)); do
    case "$1" in
      --id)
        [[ $# -ge 2 ]] || die "--id requires a value"
        detector_id=$2
        shift 2
        ;;
      --version)
        [[ $# -ge 2 ]] || die "--version requires a value"
        detector_version=$2
        shift 2
        ;;
      --activate)
        activate=true
        shift
        ;;
      --keep-admin-pod)
        KEEP_ADMIN_POD=true
        shift
        ;;
      *) die "unknown publish argument '$1'" ;;
    esac
  done

  [[ -n "${detector_id}" ]] || die "publish requires --id"
  [[ -n "${detector_version}" ]] || die "publish requires --version"
  validate_detector_id "${detector_id}"
  validate_version "${detector_version}"
  validate_local_source "${source_dir}"
  require_command python3
  prepare_admin_pod

  REMOTE_SOURCE="/opt/cloudsentinel/detectors/staging/incoming/publish_$$_${RANDOM}"
  kubectl exec -n "${NAMESPACE}" "${ADMIN_POD}" -- mkdir -p "${REMOTE_SOURCE}"
  printf 'Copying trusted plugin package...\n'
  kubectl cp "${source_dir}/." "${NAMESPACE}/${ADMIN_POD}:${REMOTE_SOURCE}"

  printf 'Validating plugin package and runtime compatibility...\n'
  local validation_json
  validation_json="$(kubectl exec -n "${NAMESPACE}" "${ADMIN_POD}" -- \
    cloudsentinel-plugin validate "${REMOTE_SOURCE}")"
  local actual_id actual_version
  actual_id="$(printf '%s' "${validation_json}" | json_field detector_id)"
  actual_version="$(printf '%s' "${validation_json}" | json_field detector_version)"
  [[ "${actual_id}" == "${detector_id}" ]] \
    || die "manifest detector ID does not match --id"
  [[ "${actual_version}" == "${detector_version}" ]] \
    || die "manifest detector version does not match --version"

  printf 'Publishing immutable plugin release...\n'
  local install_json
  install_json="$(kubectl exec -n "${NAMESPACE}" "${ADMIN_POD}" -- \
    cloudsentinel-plugin install "${REMOTE_SOURCE}")"
  actual_id="$(printf '%s' "${install_json}" | json_field detector_id)"
  actual_version="$(printf '%s' "${install_json}" | json_field detector_version)"
  [[ "${actual_id}" == "${detector_id}" && "${actual_version}" == "${detector_version}" ]] \
    || die "installed release identity does not match the requested plugin"

  if [[ "${activate}" == true ]]; then
    printf 'Activating plugin release...\n'
    kubectl exec -n "${NAMESPACE}" "${ADMIN_POD}" -- \
      cloudsentinel-plugin activate "${detector_id}" "${detector_version}" >/dev/null
    printf "Published and activated external detector '%s' version '%s'.\n" \
      "${detector_id}" "${detector_version}"
    printf 'It is now available to the Detector Library on its next refresh.\n'
  else
    printf "Published external detector '%s' version '%s' without changing the active release.\n" \
      "${detector_id}" "${detector_version}"
    printf 'Run publish again with --activate, or use the technical CLI, to activate it.\n'
  fi
}

rollback_plugin() {
  local detector_id=""
  local selector=""
  [[ $# -ge 2 ]] || die "rollback requires DETECTOR_ID and VERSION_OR_SHA256"
  detector_id=$1
  selector=$2
  shift 2
  while (($#)); do
    case "$1" in
      --keep-admin-pod)
        KEEP_ADMIN_POD=true
        shift
        ;;
      *) die "unknown rollback argument '$1'" ;;
    esac
  done
  validate_detector_id "${detector_id}"
  validate_selector "${selector}"
  prepare_admin_pod
  kubectl exec -n "${NAMESPACE}" "${ADMIN_POD}" -- \
    cloudsentinel-plugin rollback "${detector_id}" "${selector}" >/dev/null
  printf "Activated installed release '%s' for external detector '%s'.\n" \
    "${selector}" "${detector_id}"
}

list_plugins() {
  while (($#)); do
    case "$1" in
      --keep-admin-pod)
        KEEP_ADMIN_POD=true
        shift
        ;;
      *) die "unknown list argument '$1'" ;;
    esac
  done
  prepare_admin_pod
  kubectl exec -n "${NAMESPACE}" "${ADMIN_POD}" -- cloudsentinel-plugin list
}

main() {
  [[ $# -ge 1 ]] || { usage >&2; exit 2; }
  case "$1" in
    publish)
      shift
      publish_plugin "$@"
      ;;
    rollback)
      shift
      rollback_plugin "$@"
      ;;
    list)
      shift
      list_plugins "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage >&2
      die "unknown command '$1'"
      ;;
  esac
}

main "$@"
