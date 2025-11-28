"""
Test API Connectivity

Manual script to test Football-Data.org and The Odds API connectivity.
Verifies API keys, displays sample responses, and checks quota usage.
"""

import asyncio
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from src.data_collection.infrastructure.apis.football_data_client import (
    FootballDataClient,
    MatchStatus,
)
from src.data_collection.infrastructure.apis.odds_api_client import (
    OddsAPIClient,
    Sport,
    OddsRegion,
    OddsMarket,
)
from src.shared.config.settings import settings

console = Console()


async def test_football_data_api():
    """Test Football-Data.org API"""
    console.print("\n[bold cyan]Testing Football-Data.org API[/bold cyan]\n")
    
    try:
        async with FootballDataClient() as client:
            # Test 1: Get upcoming matches for La Liga
            console.print("[yellow]Fetching upcoming La Liga matches...[/yellow]")
            matches = await client.get_upcoming_matches(
                league_id=settings.league_laliga_id,
                days_ahead=7
            )
            
            # Display results in table
            table = Table(title="Upcoming La Liga Matches")
            table.add_column("Date", style="cyan")
            table.add_column("Home Team", style="green")
            table.add_column("Away Team", style="red")
            table.add_column("Status", style="yellow")
            
            for match in matches.matches[:5]:  # Show first 5
                table.add_row(
                    match.utcDate.strftime("%Y-%m-%d %H:%M"),
                    match.homeTeam.name,
                    match.awayTeam.name,
                    match.status
                )
            
            console.print(table)
            console.print(f"\n[green]✓[/green] Found {len(matches.matches)} upcoming matches")
            
            # Test 2: Get league standings
            console.print("\n[yellow]Fetching La Liga standings...[/yellow]")
            standings = await client.get_league_standings(settings.league_laliga_id)
            
            # Display top 5 teams
            table = Table(title="La Liga Standings (Top 5)")
            table.add_column("Pos", style="cyan")
            table.add_column("Team", style="green")
            table.add_column("Played", style="white")
            table.add_column("Points", style="yellow")
            table.add_column("GD", style="magenta")
            
            if standings.standings:
                for position in standings.standings[0].table[:5]:
                    table.add_row(
                        str(position.position),
                        position.team.name,
                        str(position.playedGames),
                        str(position.points),
                        str(position.goalDifference)
                    )
            
            console.print(table)
            
            # Display stats
            stats = client.get_stats()
            console.print(f"\n[bold]API Statistics:[/bold]")
            console.print(f"  Total Requests: {stats['total_requests']}")
            console.print(f"  Total Errors: {stats['total_errors']}")
            console.print(f"  Success Rate: {stats['success_rate']:.1f}%")
            
            console.print("\n[bold green]✓ Football-Data.org API test passed![/bold green]")
            return True
            
    except Exception as e:
        console.print(f"\n[bold red]✗ Football-Data.org API test failed:[/bold red]")
        console.print(f"  Error: {str(e)}")
        return False


async def test_odds_api():
    """Test The Odds API"""
    console.print("\n[bold cyan]Testing The Odds API[/bold cyan]\n")
    
    try:
        async with OddsAPIClient() as client:
            # Test 1: Get available sports
            console.print("[yellow]Fetching available sports...[/yellow]")
            sports = await client.get_sports()
            
            # Filter soccer sports
            soccer_sports = [s for s in sports if "soccer" in s.key]
            
            table = Table(title="Available Soccer Leagues")
            table.add_column("Key", style="cyan")
            table.add_column("Title", style="green")
            table.add_column("Active", style="yellow")
            
            for sport in soccer_sports[:10]:  # Show first 10
                table.add_row(
                    sport.key,
                    sport.title,
                    "✓" if sport.active else "✗"
                )
            
            console.print(table)
            console.print(f"\n[green]✓[/green] Found {len(soccer_sports)} soccer leagues")
            
            # Test 2: Get odds for Premier League
            console.print("\n[yellow]Fetching Premier League odds...[/yellow]")
            odds_response = await client.get_odds(
                sport=Sport.SOCCER_EPL,
                regions=[OddsRegion.EU],
                markets=[OddsMarket.H2H, OddsMarket.TOTALS]
            )
            
            # Display first 3 matches with odds
            table = Table(title="Premier League Odds (Sample)")
            table.add_column("Match", style="cyan")
            table.add_column("Commence Time", style="yellow")
            table.add_column("Bookmakers", style="green")
            
            for event in odds_response.events[:3]:
                table.add_row(
                    f"{event.home_team} vs {event.away_team}",
                    event.commence_time.strftime("%Y-%m-%d %H:%M"),
                    str(len(event.bookmakers))
                )
            
            console.print(table)
            
            # Show sample odds from first event
            if odds_response.events:
                first_event = odds_response.events[0]
                console.print(f"\n[bold]Sample Odds for: {first_event.home_team} vs {first_event.away_team}[/bold]")
                
                if first_event.bookmakers:
                    first_bookmaker = first_event.bookmakers[0]
                    console.print(f"  Bookmaker: {first_bookmaker.title}")
                    
                    for market in first_bookmaker.markets:
                        console.print(f"  Market: {market.key}")
                        for outcome in market.outcomes:
                            console.print(f"    {outcome.name}: {outcome.price}")
            
            # Display quota usage
            quota = client.get_quota_usage()
            console.print(f"\n[bold]API Quota Usage:[/bold]")
            console.print(f"  Used: {quota['requests_used']}/{quota['quota_limit']}")
            console.print(f"  Remaining: {quota['requests_remaining']}")
            console.print(f"  Usage: {quota['usage_percentage']}%")
            
            if quota['approaching_limit']:
                console.print(f"  [bold red]⚠️  Warning: Approaching quota limit![/bold red]")
            else:
                console.print(f"  [green]✓ Quota OK[/green]")
            
            # Display stats
            stats = client.get_stats()
            console.print(f"\n[bold]API Statistics:[/bold]")
            console.print(f"  Total Requests: {stats['total_requests']}")
            console.print(f"  Total Errors: {stats['total_errors']}")
            console.print(f"  Success Rate: {stats['success_rate']:.1f}%")
            
            console.print("\n[bold green]✓ The Odds API test passed![/bold green]")
            return True
            
    except Exception as e:
        console.print(f"\n[bold red]✗ The Odds API test failed:[/bold red]")
        console.print(f"  Error: {str(e)}")
        return False


async def main():
    """Run all API tests"""
    console.print(Panel.fit(
        "[bold cyan]Sports Betting Intelligence Platform[/bold cyan]\n"
        "[yellow]API Connectivity Test[/yellow]",
        border_style="cyan"
    ))
    
    # Test both APIs
    football_data_ok = await test_football_data_api()
    odds_api_ok = await test_odds_api()
    
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold]Test Summary:[/bold]")
    console.print(f"  Football-Data.org API: {'[green]✓ PASS[/green]' if football_data_ok else '[red]✗ FAIL[/red]'}")
    console.print(f"  The Odds API: {'[green]✓ PASS[/green]' if odds_api_ok else '[red]✗ FAIL[/red]'}")
    
    if football_data_ok and odds_api_ok:
        console.print("\n[bold green]All API tests passed! ✓[/bold green]")
        console.print("[dim]You can now proceed with data collection.[/dim]")
    else:
        console.print("\n[bold red]Some API tests failed. Please check your API keys.[/bold red]")
    
    console.print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
