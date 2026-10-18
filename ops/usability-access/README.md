# Temporary CloudSentinel usability access

This directory prepares one temporary public entry point:

```text
https://16.16.28.122
  -> Nginx HTTPS + HTTP Basic Authentication
  -> 127.0.0.1:18000
  -> systemd-managed kubectl port-forward
  -> Service/monitoring-project-service in namespace cloudsentinel
```

It contains no password, certificate, private key or AWS credential. Do not
apply these files until the security-group association and change approval have
been confirmed.

## Certificate choice

Let's Encrypt IP certificates are publicly trusted, use the mandatory
`shortlived` profile, and are valid for approximately 160 hours. Certbot 5.4+
is required for IP-address webroot issuance. Ubuntu's apt candidate is 4.0.0,
so install the current Certbot snap and verify its version before issuance.
The snap renewal timer plus the deploy hook below renews and reloads Nginx
without an interactive shell.

## Prepared installation commands (do not run before approval)

Run from the repository root on EC2. Install while ports 80 and 443 are still
closed in the security group so the package's default site is never the public
application entry point.

```bash
export REPO_ROOT="$PWD"

sudo apt-get update
sudo apt-get install --yes nginx apache2-utils
sudo snap install certbot --classic
/snap/bin/certbot --version

sudo rm -f /etc/nginx/sites-enabled/default
sudo install -d -m 0755 /var/www/letsencrypt/.well-known/acme-challenge
sudo install -m 0644 \
  "$REPO_ROOT/ops/usability-access/cloudsentinel-http-bootstrap.nginx" \
  /etc/nginx/conf.d/cloudsentinel.conf
sudo nginx -t
sudo systemctl enable --now nginx

sudo install -m 0644 \
  "$REPO_ROOT/ops/usability-access/cloudsentinel-port-forward.service" \
  /etc/systemd/system/cloudsentinel-port-forward.service
sudo systemctl daemon-reload
sudo systemctl enable --now cloudsentinel-port-forward.service
curl --fail --silent --show-error http://127.0.0.1:18000/ >/dev/null
```

After TCP 80 is allowed from `0.0.0.0/0`, exercise Let's Encrypt staging first.
Replace the email value locally; do not commit it.

```bash
read -r -p 'Lets Encrypt account email: ' LETSENCRYPT_EMAIL

sudo /snap/bin/certbot certonly \
  --staging \
  --non-interactive \
  --agree-tos \
  --email "$LETSENCRYPT_EMAIL" \
  --preferred-profile shortlived \
  --webroot \
  --webroot-path /var/www/letsencrypt \
  --ip-address 16.16.28.122 \
  --cert-name 16.16.28.122-staging

sudo /snap/bin/certbot certonly \
  --non-interactive \
  --agree-tos \
  --email "$LETSENCRYPT_EMAIL" \
  --preferred-profile shortlived \
  --webroot \
  --webroot-path /var/www/letsencrypt \
  --ip-address 16.16.28.122 \
  --cert-name 16.16.28.122

# The staging certificate has served its purpose and must not be renewed.
sudo /snap/bin/certbot delete \
  --non-interactive \
  --cert-name 16.16.28.122-staging

unset LETSENCRYPT_EMAIL
```

Generate the participant password file on the local Mac. The password is read
without echo and is not placed on a command line or in shell history. Only the
bcrypt hash is copied to EC2.

```bash
umask 077
read -r -s -p 'Participant password: ' CLOUDSENTINEL_PASSWORD; echo
printf '%s\n' "$CLOUDSENTINEL_PASSWORD" \
  | /usr/sbin/htpasswd -nBi participant \
  > /tmp/cloudsentinel.htpasswd
unset CLOUDSENTINEL_PASSWORD

scp -i /Users/sue/Documents/ThesisDevTools/boutique_baseline_project.pem \
  /tmp/cloudsentinel.htpasswd ubuntu@16.16.28.122:/tmp/cloudsentinel.htpasswd
```

Then install only the hash, the final proxy configuration and the renewal hook
on EC2. Open TCP 443 for the approved participant addresses only after the
local checks pass.

```bash
export REPO_ROOT="$PWD"
sudo install -o root -g www-data -m 0640 \
  /tmp/cloudsentinel.htpasswd /etc/nginx/cloudsentinel.htpasswd
rm -f /tmp/cloudsentinel.htpasswd

sudo install -m 0644 \
  "$REPO_ROOT/ops/usability-access/cloudsentinel-https.nginx" \
  /etc/nginx/conf.d/cloudsentinel.conf
sudo install -m 0755 \
  "$REPO_ROOT/ops/usability-access/reload-nginx-after-renewal.sh" \
  /etc/letsencrypt/renewal-hooks/deploy/reload-nginx-after-renewal.sh
sudo nginx -t
sudo systemctl reload nginx

systemctl is-active cloudsentinel-port-forward.service nginx
sudo systemctl list-timers --all | grep -E 'certbot|snap.certbot'
sudo /snap/bin/certbot renew --dry-run \
  --run-deploy-hooks
curl --resolve 16.16.28.122:443:127.0.0.1 \
  --user participant \
  --head https://16.16.28.122/
```

## Application image and Kubernetes service

The prepared Django settings accept the public IP, trust HTTPS only when the
deployment explicitly enables the proxy flag, and mark session/CSRF cookies
secure. The Service becomes ClusterIP, removing NodePort 30000 while preserving
`kubectl port-forward`.

```bash
docker build \
  -f user_interface/monitoring_project/Dockerfile \
  -t jojojochem/monitoring_project:usability-access-1 \
  user_interface/monitoring_project

minikube image load jojojochem/monitoring_project:usability-access-1
kubectl apply -f k8s/monitoring_project-deployment.yml
kubectl rollout status deployment/monitoring-project-deployment \
  -n cloudsentinel --timeout=180s
kubectl get service monitoring-project-service -n cloudsentinel -o wide
```

The expected Service type is `ClusterIP`; no security-group rule may expose
TCP 30000.

## AWS security group

In AWS Console, open **EC2 -> Instances -> i-019f93344daf4e90a -> Security ->
sg-0c5a9e3150f0ee011**. First open **Associated resources**. Continue only if
the group is attached solely to this instance. If other instances appear, do
not edit it until their exposure has been reviewed.

Add these IPv4 inbound rules:

| Type | Protocol | Port | Source | Purpose |
| --- | --- | ---: | --- | --- |
| HTTP | TCP | 80 | `0.0.0.0/0` | Let's Encrypt HTTP-01 only; all other paths return 404 |
| HTTPS | TCP | 443 | each approved IPv4 as `x.x.x.x/32` | One rule per participant/admin address |

Keep the existing SSH/TCP 22 rule and its current source unchanged. Do not add
TCP 30000. If a 30000 rule already exists, remove that rule only after checking
that it is not shared with another instance/workload.

Do not add an HTTPS `0.0.0.0/0` or IPv6 `::/0` rule. In the AWS rule editor,
choose **Custom** as the HTTPS source and enter the participant's current public
IPv4 with `/32`. **My IP** may be used for the administrator's current address.
Mobile and residential addresses can change; update the corresponding `/32`
rule before a session rather than widening it.

Port 80 cannot be restricted to a stable Let's Encrypt allowlist: Let's Encrypt
does not publish its validation ranges and they may change. Nginx exposes only
the ACME challenge directory on that port; application traffic is never proxied
over HTTP.

## Non-destructive shutdown and rollback

First remove the inbound 80/443 rules added for the evaluation. Then stop the
temporary host services; stored certificates and hashes remain available for
later sessions.

```bash
sudo systemctl disable --now nginx
sudo systemctl disable --now cloudsentinel-port-forward.service
```

To restore only the prior Kubernetes service shape, revert the repository
manifest to NodePort and apply it. This is not needed for normal SSH plus
`kubectl port-forward` access.
