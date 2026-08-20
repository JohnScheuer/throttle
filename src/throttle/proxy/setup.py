"""Setup command to configure Claude Code to use Throttle proxy."""
import json
import os
import sys
from getpass import getpass
from pathlib import Path
from typing import Optional


def get_claude_settings_path() -> Path:
    """Get the path to Claude Code settings.json."""
    return Path.home() / ".claude" / "settings.json"


def read_settings() -> dict:
    """Read existing Claude Code settings or return empty dict."""
    settings_path = get_claude_settings_path()
    
    if not settings_path.exists():
        return {}
    
    try:
        with open(settings_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def write_settings(settings: dict) -> None:
    """Write settings to Claude Code settings.json."""
    settings_path = get_claude_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
    
    print(f"✅ Settings written to {settings_path}")


def save_api_key(api_key: str) -> None:
    """Save API key to Throttle config."""
    config_dir = Path.home() / ".throttle"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / "config.json"
    config = {}
    
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError:
            config = {}
    
    config["api_key"] = api_key
    
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    # Set restrictive permissions
    config_file.chmod(0o600)
    print(f"✅ API key saved to {config_file}")


def load_api_key() -> Optional[str]:
    """Load API key from Throttle config."""
    config_file = Path.home() / ".throttle" / "config.json"
    
    if not config_file.exists():
        return None
    
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
            return config.get("api_key")
    except (json.JSONDecodeError, OSError):
        return None


def setup(port: int = 8080) -> None:
    """Run the setup wizard."""
    print("🔧 Throttle Setup")
    print("=" * 50)
    print()
    print("This will configure Claude Code to use Throttle as a")
    print("cost-tracking and caching layer for the Anthropic API.")
    print()
    
    # Get API key
    print("Step 1: Anthropic API Key")
    print("-" * 50)
    print("Enter your Anthropic Console API key.")
    print("(This is NOT your claude.ai login - it's the API key")
    print("from https://console.anthropic.com/settings/keys)")
    print()
    
    existing_key = load_api_key()
    if existing_key:
        print(f"Found existing API key: {existing_key[:8]}...{existing_key[-4:]}")
        use_existing = input("Use existing key? [Y/n]: ").strip().lower()
        if use_existing in ("", "y", "yes"):
            api_key = existing_key
        else:
            api_key = getpass("API Key: ").strip()
    else:
        api_key = getpass("API Key: ").strip()
    
    if not api_key or not api_key.startswith("sk-ant-"):
        print("❌ Invalid API key format. Should start with 'sk-ant-'")
        sys.exit(1)
    
    save_api_key(api_key)
    
    # Set environment variable for the proxy to use
    os.environ["ANTHROPIC_API_KEY"] = api_key
    
    print()
    print("Step 2: Configure Claude Code")
    print("-" * 50)
    
    settings = read_settings()
    
    # Ensure env block exists
    if "env" not in settings:
        settings["env"] = {}
    
    # Set proxy configuration
    proxy_url = f"http://localhost:{port}"
    settings["env"]["ANTHROPIC_BASE_URL"] = proxy_url
    settings["env"]["ANTHROPIC_API_KEY"] = api_key
    
    write_settings(settings)
    
    print()
    print("✅ Setup complete!")
    print()
    print("Next steps:")
    print(f"  1. Start the proxy: throttle-proxy {port}")
    print("  2. Restart Claude Code")
    print("  3. Use Claude Code normally - all requests will be logged and cached")
    print()
    print(f"View stats anytime with: throttle-summary")
    print()


def main() -> None:
    """Entry point for setup command."""
    port = int(os.getenv("THROTTLE_PORT", "8080"))
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ("-h", "--help"):
            print("Usage: throttle-setup [PORT]")
            print()
            print("Configure Claude Code to use Throttle proxy.")
            print(f"Default port: {port}")
            sys.exit(0)
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}", file=sys.stderr)
            sys.exit(1)
    
    try:
        setup(port)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
