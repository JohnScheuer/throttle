# Throttle Cost Layer - Quick Start Guide

Throttle now includes a **cost-tracking and caching HTTP proxy** for Claude Code that:
- Logs every API request (tokens, latency, cost)
- Caches non-tool-use responses to save money
- Provides usage statistics

## Setup (One Time)

1. **Run setup** to configure your Anthropic API key and Claude Code:
   ```bash
   throttle-setup
   ```
   
   This will:
   - Prompt for your Anthropic Console API key (from https://console.anthropic.com/settings/keys)
   - Save it securely to `~/.throttle/config.json`
   - Configure `~/.claude/settings.json` to use the proxy

2. **Start the proxy** (keep it running):
   ```bash
   throttle-proxy
   ```
   
   You should see:
   ```
   🚀 Throttle proxy starting on http://localhost:8080
   📁 Logs: /Users/you/.throttle/logs
   💾 Cache: /Users/you/.throttle/cache
   🔑 API key: configured
   
   Forwarding requests to https://api.anthropic.com
   ```

3. **Restart Claude Code** so it picks up the new proxy configuration.

## Usage

Just use Claude Code normally. The proxy runs in the background and:
- Forwards all requests to Anthropic unchanged
- Logs tokens and cost for every request
- Caches responses that don't contain tool use (file edits, commands)
- Returns cached responses when safe (same request, unchanged files)

## View Statistics

Run anytime to see your usage:
```bash
throttle-summary
```

Example output:
```
📊 Throttle Usage Summary
======================================================================

Today
----------------------------------------------------------------------
  Requests:        47
  Cache hits:      12 (25.5%)
  Cache misses:    35

  Input tokens:    125,430
  Output tokens:   18,920
  Cache reads:     45,200
  Cache writes:    12,100

  💰 Cost:         $0.8234
  💚 Saved:        $0.2145
  📈 Savings:      20.7%

Last 7 Days
----------------------------------------------------------------------
  Requests:        312
  Cache hits:      89 (28.5%)
  Cache misses:    223

  💰 Cost:         $5.67
  💚 Saved:        $1.89
  📈 Savings:      25.0%
```

## How Caching Works

**Cached:**
- Pure informational responses (explanations, code reviews, etc.)
- Identical requests within 24 hours
- Only if referenced files haven't changed

**Not Cached:**
- Responses with tool use (file edits, bash commands)
- Requests with changed file dependencies
- Responses older than 24 hours

This prevents cache corruption while maximizing savings.

## Files

- `~/.throttle/config.json` - API key (mode 0600)
- `~/.throttle/logs/throttle-YYYY-MM-DD.jsonl` - Daily request logs
- `~/.throttle/cache/*.json` - Cached responses
- `~/.claude/settings.json` - Claude Code configuration

## Stopping

Press `Ctrl+C` in the terminal running `throttle-proxy`.

To disable the proxy entirely, edit `~/.claude/settings.json` and remove the `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` entries from the `env` block, then restart Claude Code.

## Custom Port

Use a different port:
```bash
throttle-setup 9000
throttle-proxy 9000
```

## Troubleshooting

**"ERROR: ANTHROPIC_API_KEY not found"**
- Run `throttle-setup` first

**Claude Code not using proxy**
- Check `~/.claude/settings.json` has `ANTHROPIC_BASE_URL` set
- Restart Claude Code after running setup
- Ensure proxy is running (`throttle-proxy`)

**No cache hits**
- Cache only works for non-tool-use responses
- Files must be unchanged since last request
- Request must be identical (including system prompt)
