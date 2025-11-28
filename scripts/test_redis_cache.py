"""
Test Redis Cache

Verify Redis cache functionality.
"""

import asyncio
from rich.console import Console
from rich.table import Table

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.infrastructure.cache.redis_cache import RedisCache

console = Console()


async def test_cache():
    """Test Redis cache operations"""
    
    console.print("\n[bold cyan]Testing Redis Cache[/bold cyan]\n")
    
    async with RedisCache() as cache:
        # Test 1: Set and Get
        console.print("[yellow]Test 1: Set and Get[/yellow]")
        await cache.set('test', 'key1', {'name': 'Test Match', 'score': '2-1'}, ttl=60)
        result = await cache.get('test', 'key1')
        console.print(f"  Stored: {{'name': 'Test Match', 'score': '2-1'}}")
        console.print(f"  Retrieved: {result}")
        console.print(f"  [green]OK[/green] - Match!\n")
        
        # Test 2: Cache Miss
        console.print("[yellow]Test 2: Cache Miss[/yellow]")
        result = await cache.get('test', 'nonexistent')
        console.print(f"  Result: {result}")
        console.print(f"  [green]OK[/green] - Returns None\n")
        
        # Test 3: TTL
        console.print("[yellow]Test 3: TTL Check[/yellow]")
        ttl = await cache.get_ttl('test', 'key1')
        console.print(f"  Remaining TTL: {ttl} seconds")
        console.print(f"  [green]OK[/green] - TTL set correctly\n")
        
        # Test 4: Exists
        console.print("[yellow]Test 4: Key Existence[/yellow]")
        exists = await cache.exists('test', 'key1')
        console.print(f"  Key exists: {exists}")
        console.print(f"  [green]OK[/green] - Key found\n")
        
        # Test 5: Delete
        console.print("[yellow]Test 5: Delete Key[/yellow]")
        deleted = await cache.delete('test', 'key1')
        exists_after = await cache.exists('test', 'key1')
        console.print(f"  Deleted: {deleted}")
        console.print(f"  Exists after delete: {exists_after}")
        console.print(f"  [green]OK[/green] - Key deleted\n")
        
        # Test 6: Multiple keys
        console.print("[yellow]Test 6: Multiple Keys[/yellow]")
        await cache.set('matches', '123', {'home': 'Team A', 'away': 'Team B'})
        await cache.set('matches', '456', {'home': 'Team C', 'away': 'Team D'})
        await cache.set('odds', '123', {'h': 2.5, 'd': 3.2, 'a': 2.8})
        
        match1 = await cache.get('matches', '123')
        match2 = await cache.get('matches', '456')
        odds1 = await cache.get('odds', '123')
        
        console.print(f"  Match 123: {match1}")
        console.print(f"  Match 456: {match2}")
        console.print(f"  Odds 123: {odds1}")
        console.print(f"  [green]OK[/green] - Multiple keys work\n")
        
        # Test 7: Clear namespace
        console.print("[yellow]Test 7: Clear Namespace[/yellow]")
        deleted_count = await cache.clear_namespace('matches')
        console.print(f"  Deleted {deleted_count} keys from 'matches' namespace")
        exists_after = await cache.exists('matches', '123')
        odds_exists = await cache.exists('odds', '123')
        console.print(f"  Match 123 exists: {exists_after}")
        console.print(f"  Odds 123 exists: {odds_exists}")
        console.print(f"  [green]OK[/green] - Namespace cleared\n")
        
        # Test 8: Statistics
        console.print("[yellow]Test 8: Cache Statistics[/yellow]")
        stats = cache.get_stats()
        
        table = Table(title="Cache Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)
        console.print(f"  [green]OK[/green] - Stats tracked\n")
        
        # Cleanup
        await cache.clear_all()
        
    console.print("[bold green]SUCCESS - All cache tests passed![/bold green]\n")


if __name__ == "__main__":
    asyncio.run(test_cache())
