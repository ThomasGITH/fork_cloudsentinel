#!/bin/sh
set -eu

/usr/sbin/nginx -t >/dev/null 2>&1
/bin/systemctl reload nginx
