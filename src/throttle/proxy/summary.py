"""Summary command to display usage statistics from Throttle logs."""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple


def get_log_dir() -> Path:
    """Get the Throttle logs directory."""
    return Path.home() / ".throttle" / "logs"


def parse_log_file(log_file: Path) -> List[Dict]:
    """Parse a JSONL log file and return entries."""
    entries = []
    if not log_file.exists():
        return entries
    
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return entries


def calculate_stats(entries: List[Dict]) -> Dict:
    """Calculate statistics from log entries."""
    total_requests = len(entries)
    cache_hits = sum(1 for e in entries if e.get("cache_hit"))
    cache_misses = total_requests - cache_hits
    
    total_cost = sum(e.get("cost_usd", 0.0) for e in entries)
    total_saved = sum(e.get("saved_usd", 0.0) for e in entries)
    
    cache_hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0.0
    
    # Token stats
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read_tokens = 0
    total_cache_write_tokens = 0
    
    for entry in entries:
        usage = entry.get("usage", {})
        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)
        total_cache_read_tokens += usage.get("cache_read_input_tokens", 0)
        total_cache_write_tokens += usage.get("cache_creation_input_tokens", 0)
    
    return {
        "total_requests": total_requests,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hit_rate,
        "total_cost": total_cost,
        "total_saved": total_saved,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cache_read_tokens": total_cache_read_tokens,
        "total_cache_write_tokens": total_cache_write_tokens,
    }


def get_date_range_stats(days: int) -> Tuple[Dict, List[Tuple[str, Dict]]]:
    """Get statistics for the last N days."""
    log_dir = get_log_dir()
    today = datetime.now().date()
    
    all_entries = []
    daily_stats = []
    
    for i in range(days):
        date = today - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        log_file = log_dir / f"throttle-{date_str}.jsonl"
        
        entries = parse_log_file(log_file)
        all_entries.extend(entries)
        
        if entries:
            stats = calculate_stats(entries)
            daily_stats.append((date_str, stats))
    
    total_stats = calculate_stats(all_entries)
    daily_stats.reverse()  # Oldest to newest
    
    return total_stats, daily_stats


def format_currency(amount: float) -> str:
    """Format currency with appropriate precision."""
    if amount >= 1.0:
        return f"${amount:.2f}"
    elif amount >= 0.01:
        return f"${amount:.3f}"
    else:
        return f"${amount:.4f}"


def format_number(num: int) -> str:
    """Format large numbers with commas."""
    return f"{num:,}"


def print_summary() -> None:
    """Print usage summary for today and last 7 days."""
    print("📊 Throttle Usage Summary")
    print("=" * 70)
    print()
    
    # Today's stats
    print("Today")
    print("-" * 70)
    today_stats, _ = get_date_range_stats(1)
    
    if today_stats["total_requests"] == 0:
        print("No requests logged today.")
    else:
        print(f"  Requests:        {format_number(today_stats['total_requests'])}")
        print(f"  Cache hits:      {format_number(today_stats['cache_hits'])} ({today_stats['cache_hit_rate']:.1f}%)")
        print(f"  Cache misses:    {format_number(today_stats['cache_misses'])}")
        print()
        print(f"  Input tokens:    {format_number(today_stats['total_input_tokens'])}")
        print(f"  Output tokens:   {format_number(today_stats['total_output_tokens'])}")
        print(f"  Cache reads:     {format_number(today_stats['total_cache_read_tokens'])}")
        print(f"  Cache writes:    {format_number(today_stats['total_cache_write_tokens'])}")
        print()
        print(f"  💰 Cost:         {format_currency(today_stats['total_cost'])}")
        print(f"  💚 Saved:        {format_currency(today_stats['total_saved'])}")
        
        if today_stats['total_cost'] + today_stats['total_saved'] > 0:
            total = today_stats['total_cost'] + today_stats['total_saved']
            savings_pct = (today_stats['total_saved'] / total) * 100
            print(f"  📈 Savings:      {savings_pct:.1f}%")
    
    print()
    print()
    
    # Last 7 days
    print("Last 7 Days")
    print("-" * 70)
    week_stats, daily_breakdown = get_date_range_stats(7)
    
    if week_stats["total_requests"] == 0:
        print("No requests logged in the last 7 days.")
    else:
        print(f"  Requests:        {format_number(week_stats['total_requests'])}")
        print(f"  Cache hits:      {format_number(week_stats['cache_hits'])} ({week_stats['cache_hit_rate']:.1f}%)")
        print(f"  Cache misses:    {format_number(week_stats['cache_misses'])}")
        print()
        print(f"  Input tokens:    {format_number(week_stats['total_input_tokens'])}")
        print(f"  Output tokens:   {format_number(week_stats['total_output_tokens'])}")
        print(f"  Cache reads:     {format_number(week_stats['total_cache_read_tokens'])}")
        print(f"  Cache writes:    {format_number(week_stats['total_cache_write_tokens'])}")
        print()
        print(f"  💰 Cost:         {format_currency(week_stats['total_cost'])}")
        print(f"  💚 Saved:        {format_currency(week_stats['total_saved'])}")
        
        if week_stats['total_cost'] + week_stats['total_saved'] > 0:
            total = week_stats['total_cost'] + week_stats['total_saved']
            savings_pct = (week_stats['total_saved'] / total) * 100
            print(f"  📈 Savings:      {savings_pct:.1f}%")
        
        # Daily breakdown if there's data
        if len(daily_breakdown) > 1:
            print()
            print("  Daily Breakdown:")
            print("  " + "-" * 66)
            for date_str, stats in daily_breakdown:
                if stats["total_requests"] > 0:
                    print(f"  {date_str}  {stats['total_requests']:4d} req  "
                          f"{stats['cache_hit_rate']:5.1f}% hit  "
                          f"cost {format_currency(stats['total_cost']):>8s}  "
                          f"saved {format_currency(stats['total_saved']):>8s}")
    
    print()


def main() -> None:
    """Entry point for summary command."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: throttle-summary")
        print()
        print("Display Throttle usage statistics for today and the last 7 days.")
        sys.exit(0)
    
    try:
        print_summary()
    except Exception as e:
        print(f"Error generating summary: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
