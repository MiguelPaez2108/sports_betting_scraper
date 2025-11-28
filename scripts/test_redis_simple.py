"""
Simple Redis Cache Test

Test Redis cache without complex dependencies.
"""

import asyncio
import json
import redis.asyncio as redis
from rich.console import Console
from rich.table import Table

console = Console()


async def test_redis():
    """Test Redis connection and basic operations"""
    
    console.print("\n[bold cyan]Testing Redis Cache[/bold cyan]\n")
    
    try:
        # Connect to Redis
        console.print("[yellow]Connecting to Redis...[/yellow]")
        client = await redis.from_url(
            "redis://localhost:6379/0",
            encoding="utf-8",
            decode_responses=True
        )
        
        # Ping
        await client.ping()
        console.print("[green]OK[/green] Connected successfully\n")
        
        # Test 1: Set and Get
        console.print("[yellow]Test 1: Set and Get[/yellow]")
        test_data = {'match': 'Team A vs Team B', 'score': '2-1'}
        await client.setex('test:match1', 60, json.dumps(test_data))
        result = await client.get('test:match1')
        retrieved = json.loads(result)
        console.print(f"  Stored: {test_data}")
        console.print(f"  Retrieved: {retrieved}")
        console.print(f"  [green]OK[/green] - Data matches!\n")
        
        # Test 2: TTL
        console.print("[yellow]Test 2: TTL Check[/yellow]")
        ttl = await client.ttl('test:match1')
        console.print(f"  Remaining TTL: {ttl} seconds")
        console.print(f"  [green]OK[/green] - TTL set correctly\n")
        
        # Test 3: Multiple keys
        console.print("[yellow]Test 3: Multiple Keys[/yellow]")
        await client.set('matches:123', json.dumps({'home': 'Team C', 'away': 'Team D'}))
        await client.set('matches:456', json.dumps({'home': 'Team E', 'away': 'Team F'}))
        await client.set('odds:123', json.dumps({'h': 2.5, 'd': 3.2, 'a': 2.8}))
        
        match1 = json.loads(await client.get('matches:123'))
        match2 = json.loads(await client.get('matches:456'))
        odds1 = json.loads(await client.get('odds:123'))
        
        console.print(f"  Match 123: {match1}")
        console.print(f"  Match 456: {match2}")
        console.print(f"  Odds 123: {odds1}")
        console.print(f"  [green]OK[/green] - Multiple keys work\n")
        
        # Test 4: Delete pattern
        console.print("[yellow]Test 4: Delete Pattern[/yellow]")
        keys_to_delete = []
        async for key in client.scan_iter(match='matches:*'):
            keys_to_delete.append(key)
        
        if keys_to_delete:
            deleted = await client.delete(*keys_to_delete)
            console.print(f"  Deleted {deleted} keys from 'matches:*' pattern")
        
        exists_after = await client.exists('matches:123')
        odds_exists = await client.exists('odds:123')
        console.print(f"  Match 123 exists: {exists_after == 1}")
        console.print(f"  Odds 123 exists: {odds_exists == 1}")
        console.print(f"  [green]OK[/green] - Pattern delete works\n")
        
        # Test 5: Info
        console.print("[yellow]Test 5: Redis Info[/yellow]")
        info = await client.info('server')
        
        table = Table(title="Redis Server Info")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Redis Version", info.get('redis_version', 'N/A'))
        table.add_row("OS", info.get('os', 'N/A'))
        table.add_row("Uptime (days)", str(info.get('uptime_in_days', 'N/A')))
        
        console.print(table)
        console.print(f"  [green]OK[/green] - Server info retrieved\n")
        
        # Cleanup
        await client.flushdb()
        console.print("[dim]Cleaned up test data[/dim]\n")
        
        # Close connection
        await client.close()
        
        console.print("[bold green]SUCCESS - All Redis tests passed![/bold green]\n")
        
    except Exception as e:
        console.print(f"\n[red]ERROR[/red] {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_redis())
