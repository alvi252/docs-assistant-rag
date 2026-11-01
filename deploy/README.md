# Deploying behind a password

Docker Compose runs two containers: the assistant (not reachable from outside) and
[Caddy](https://caddyserver.com/), which asks for a login and a password before passing any
request on, and fetches an HTTPS certificate once you give it a domain. The index lives in a
Docker volume, so it survives restarts.

You need Docker with the compose plugin and an OpenAI API key. Run everything from `deploy/`.

## First start

```bash
cd deploy
cp .env.example .env
docker run --rm -it caddy:2 caddy hash-password      # type the password twice, copy the hash
```

Paste the hash into `.env` as `BASIC_AUTH_HASH='...'` (keep the single quotes) and set
`BASIC_AUTH_USER`. Put your documents (`.md`, `.txt`, `.pdf`) into `deploy/docs/`, or point
`DOCS_PATH` in `.env` at another folder. Then:

```bash
export OPENAI_API_KEY=...          # or add it to your shell profile; it is not stored in a file
docker compose up -d --build
docker compose ps                  # both containers should be Up, the app "healthy"
```

The app indexes the documents on every start; only new or changed files cost anything.
Open `http://<server>/` (port `HTTP_PORT`, 80 by default) and log in. To change the password,
make a new hash, replace it in `.env` and run `docker compose up -d`.

## HTTPS on your domain

1. Create a DNS `A` record for the domain (for example `docs.example.com`) that points to the
   server's IP address.
2. Ports 80 and 443 must be open on the server and free (`HTTP_PORT` and `HTTPS_PORT` stay at
   their defaults).
3. Set `SITE_ADDRESS=docs.example.com` in `.env` and run `docker compose up -d`.

Caddy gets and renews the certificate by itself; the certificates sit in the `caddy_data`
volume. Without a domain the chat runs over plain HTTP, so the password travels unencrypted:
use it only inside a trusted network.

## Adding documents and re-indexing

Copy the new or changed files into the documents folder, then:

```bash
docker compose exec app docs-assistant ingest             # new and changed files only
docker compose exec app docs-assistant ingest --rebuild   # re-process everything
```

Removed files disappear from the index on the next run. Restarting the stack does the same.

## Reviewing the questions asked

Questions are written to `/app/var/questions.jsonl` inside the volume (one JSON line each: time,
question, refused or not and why, cited documents, latency; no IP address, no user name). The
questions may contain personal data and never leave this server. Set `QUESTION_LOG_PATH=` (empty)
in `.env` to switch it off.

```bash
docker compose exec app docs-assistant questions              # all questions
docker compose exec app docs-assistant questions --refused    # the ones it could not answer
docker compose exec -T app docs-assistant questions --json    # one JSON object per line
```

## Stopping

```bash
docker compose stop            # stop, keep everything
docker compose down            # remove the containers, keep the index and certificates
docker compose down -v         # also delete the index, the question log and the certificates
```

Logs: `docker compose logs -f app caddy`.
