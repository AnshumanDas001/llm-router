# Deploying ThriftLLM on Oracle Cloud Always Free

End to end, on the Ampere A1 (ARM) Always Free tier. Roughly 30 minutes once
the VM exists; getting the VM is the unpredictable part.

The two places people get stuck are marked **⚠**.

## 1. Get the VM

1. Sign up at cloud.oracle.com. A card is required; Always Free resources are
   not billed. **⚠ Your home region is fixed at signup and cannot be changed.**
   Popular regions (Ashburn, Frankfurt, London, Mumbai, Singapore) frequently
   have no A1 capacity. Pick a less busy one if you don't need low latency.
2. Compute → Instances → Create instance:
   - **Image:** Canonical Ubuntu 24.04 (the *aarch64* build — it's selected
     automatically once you pick the ARM shape).
   - **Shape:** Ampere → `VM.Standard.A1.Flex`, **2 OCPUs, 12 GB RAM**. The
     free allowance is 4 OCPU / 24 GB total; leaving half unused means you can
     add a second VM later. Bump to 4/24 if you plan to run Ollama.
   - **Networking:** default VCN, assign a public IPv4.
   - **SSH key:** upload your public key.
   - **Boot volume:** 50 GB is plenty (free allowance is 200 GB total).
3. **⚠ "Out of host capacity"** on create is normal. Retry; it can take
   hours or days. People script it. Trying a different availability domain in
   the same region sometimes works immediately.
4. Note the **public IP** once it's running. You'll need it twice.

## 2. Open ports 80 and 443 — in *both* places

Oracle blocks inbound traffic at two independent layers. Opening one and
forgetting the other is the single most common "my app doesn't load" cause.

**Layer 1 — VCN security list** (cloud console):
Networking → Virtual Cloud Networks → your VCN → Security Lists → Default →
Add Ingress Rules:

| Source CIDR | Protocol | Dest port |
|---|---|---|
| `0.0.0.0/0` | TCP | 80 |
| `0.0.0.0/0` | TCP | 443 |
| `0.0.0.0/0` | UDP | 443 (HTTP/3, optional) |

**⚠ Layer 2 — iptables on the instance.** Ubuntu images from Oracle ship with
a restrictive firewall that isn't `ufw`. SSH in and run:

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p udp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

The `-I INPUT 6` inserts before Oracle's default REJECT rule. Verify with
`sudo iptables -L INPUT -n --line-numbers` — your ACCEPT lines must appear
above the `REJECT all` line.

## 3. Install Docker

```bash
ssh ubuntu@YOUR_PUBLIC_IP

sudo apt-get update && sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker ubuntu
```

Log out and back in so the group change applies, then `docker ps` should
work without sudo.

## 4. Configure and launch

```bash
git clone https://github.com/AnshumanDas001/llm-router.git thriftllm
cd thriftllm
cp .env.example .env
nano .env
```

Fill in:

```dotenv
OPENROUTER_API_KEY=...       # cheap, mid and frontier tiers; needs a few $ of credit
GROQ_API_KEY=...             # the judge; free tier is enough
ROUTER_SECRET_KEY=...        # python3 -c "import secrets; print(secrets.token_urlsafe(32))"
SITE_ADDRESS=129-146-1-2.sslip.io    # your public IP, dots -> dashes
COOKIE_SECURE=1
```

Leave `ALLOW_ANON_V1` unset on a public box: it opens the OpenAI-compatible
endpoint to anyone, and every request there spends your OpenRouter credit.

`ROUTER_SECRET_KEY` encrypts any provider keys users choose to save. Generate
it once and never change it casually — rotating it makes saved keys
unreadable.

Then:

```bash
docker compose up -d --build
```

The first build takes 5–10 minutes on A1 (installing torch). Watch it:

```bash
docker compose logs -f app
```

You want to see `Application startup complete`, then within ~15 s the
classifier warmup finishes silently. Caddy will obtain a certificate on the
first request to the hostname; give it a minute.

Open `https://129-146-1-2.sslip.io` (your address). You should get the
landing page with a valid padlock.

## 5. What it costs to run

The built-in stack is entirely hosted, so the VM does no model work — it
classifies, verifies and streams. Measured on the eval set:

| | per question |
|---|---|
| answered by the cheap tier (78% of questions) | ~$0.00006 including the judge verdict |
| escalated to mid (22%) | ~$0.005 |
| blended | ~$0.0011 |

The app caps signed-in accounts at `DAILY_PROMPT_LIMIT` prompts per day
(default 10) and the signed-out demo at `DEMO_PROMPT_LIMIT` per device
(default 3), so a stranger can cost you at most a few cents. `docker compose
logs app` shows every cascade with its cost; the Sessions page totals them.

Ollama is no longer needed. If you want the cheap tier local anyway (A1 has
the memory for a 3B model), `docker compose --profile ollama up -d --build`,
`docker compose exec ollama ollama pull llama3.2:3b`, set
`CHEAP_MODEL=ollama/llama3.2:3b`, then run `scripts/calibrate_builtin.py`
inside the container so routing uses that model's real numbers. Expect
10 s+ per cheap answer on A1's CPU; `direct` mode skips it.

If you change `CHEAP_MODEL` and want the learned gate in front of the judge
(`VERIFIER=auto`), retrain it for that model: `scripts/build_scorer_data.py`
then `scripts/train_scorer.py`. On the default stack the judge is cheap
enough that the gate isn't worth its own calls, so it's off.

Don't set a tier to a model litellm can't price (Groq's `compound-mini`,
for instance): it reports $0 per call and every cost figure in the app
becomes fiction.

## Checking the deploy is durable

`GET /health` reports where data is going, without exposing any secret:

```json
{"status": "ok", "database": "turso", "durable": true, "warnings": []}
```

If it says `"database": "sqlite"` on a container host, **accounts and chats
will be lost the next time the instance is recycled** — which presents as
"I signed up, logged in fine, and the next day my login stopped working".
The `warnings` array says which part is missing. Set `TURSO_DATABASE_URL`
and `TURSO_AUTH_TOKEN`, redeploy, and confirm `durable` is `true`.

To move an existing local database up first:

```bash
./venv/bin/python scripts/migrate_to_turso.py --dry-run   # report only
./venv/bin/python scripts/migrate_to_turso.py
```

Note `libsql` has no wheel for Linux **arm64** — on an Ampere A1 it would
have to build from Rust source. x86 hosts (Cloud Run, GCP e2-micro, Oracle
E2.1.Micro) install it from a wheel.

## Updating

```bash
cd ~/thriftllm && git pull && docker compose up -d --build
```

Data lives in Docker volumes (`router-data`, `caddy-data`), not the
container, so this is safe. To back up the database:

```bash
docker compose cp app:/app/logs/router.db ./router-backup-$(date +%F).db
```

## After it's up

- **Keep it from being reclaimed.** Oracle reclaims Always Free VMs that stay
  under ~20% CPU/memory/network for a week. Either upgrade the account to
  Pay As You Go (still $0 for Always Free resources — it only removes the
  reclamation rule) or accept the risk. Most people upgrade.
- **Rate limiting.** Nothing limits login attempts or the `/try` cap beyond a
  cookie. Cheapest fix: put Cloudflare's free tier in front (requires a
  domain) or build Caddy with the `caddy-ratelimit` plugin via `xcaddy`.
- **Memory.** `docker stats` should show the app around 1 GB (the embedding
  classifier). If it climbs toward the 2 GB limit under load, that's
  concurrent cascades holding response buffers — expected, and the limit
  will contain it.

## Troubleshooting

| symptom | almost always |
|---|---|
| Browser times out, no response at all | Layer 2 iptables (§2). Check `sudo iptables -L INPUT -n`. |
| Caddy logs `no certificate available` / cert errors | `SITE_ADDRESS` doesn't resolve to this IP, or port 80 is blocked (LE needs it for the challenge). |
| App container restarts repeatedly | `docker compose logs app` — usually a missing key in `.env`. |
| First prompt hangs ~15 s then works | Classifier warmup hadn't finished. Wait for `start_period`. |
| Streaming answers appear all at once | Something between browser and app is buffering. Caddy's `flush_interval -1` handles its side; a CDN in front may need streaming enabled. |
| `exec format error` on build | Building an amd64 image on ARM. Don't pull prebuilt x86 images; `--build` on the box builds native arm64. |
