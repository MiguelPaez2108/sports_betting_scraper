"""
Betfair Exchange Connector.

Handles:
- Authentication
- Placing bets
- Checking bet status
- Rate limiting
- Retry logic
"""
import requests
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import os


class BetStatus(Enum):
    """Bet status enum."""
    PENDING = "PENDING"
    MATCHED = "MATCHED"
    UNMATCHED = "UNMATCHED"
    CANCELLED = "CANCELLED"
    LAPSED = "LAPSED"


@dataclass
class BetOrder:
    """Bet order data."""
    market_id: str
    selection_id: str
    side: str  # 'BACK' or 'LAY'
    price: float
    size: float
    bet_id: Optional[str] = None
    status: BetStatus = BetStatus.PENDING


class BetfairConnector:
    """
    Connector for Betfair Exchange API.
    
    Supports both simulation and live modes.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        session_token: Optional[str] = None,
        simulation_mode: bool = True
    ):
        """
        Args:
            api_key: Betfair API key
            session_token: Betfair session token
            simulation_mode: If True, simulate bets without real API calls
        """
        self.api_key = api_key or os.getenv('BETFAIR_API_KEY')
        self.session_token = session_token
        self.simulation_mode = simulation_mode
        
        self.base_url = "https://api.betfair.com/exchange/betting/rest/v1.0/"
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        self.simulated_bets: Dict[str, BetOrder] = {}
        
    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(
        self,
        endpoint: str,
        method: str = 'POST',
        data: Optional[Dict] = None
    ) -> Dict:
        """Make API request with retry logic."""
        if self.simulation_mode:
            return {'status': 'SUCCESS', 'simulated': True}
        
        self._rate_limit()
        
        headers = {
            'X-Application': self.api_key,
            'X-Authentication': self.session_token,
            'Content-Type': 'application/json'
        }
        
        url = self.base_url + endpoint
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if method == 'POST':
                    response = requests.post(url, json=data, headers=headers, timeout=10)
                else:
                    response = requests.get(url, headers=headers, timeout=10)
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return {}
    
    def place_bet(
        self,
        market_id: str,
        selection_id: str,
        side: str,
        price: float,
        size: float
    ) -> BetOrder:
        """
        Place a bet on Betfair.
        
        Args:
            market_id: Betfair market ID
            selection_id: Selection ID (runner)
            side: 'BACK' or 'LAY'
            price: Decimal odds
            size: Stake amount
            
        Returns:
            BetOrder with bet details
        """
        order = BetOrder(
            market_id=market_id,
            selection_id=selection_id,
            side=side,
            price=price,
            size=size
        )
        
        if self.simulation_mode:
            # Simulate bet placement
            import uuid
            bet_id = f"SIM_{uuid.uuid4().hex[:8]}"
            order.bet_id = bet_id
            order.status = BetStatus.MATCHED
            self.simulated_bets[bet_id] = order
            
            print(f"✓ [SIMULATION] Bet placed: {bet_id} - {side} {size}@{price}")
            return order
        
        # Real API call
        data = {
            'marketId': market_id,
            'instructions': [{
                'selectionId': selection_id,
                'handicap': '0',
                'side': side,
                'orderType': 'LIMIT',
                'limitOrder': {
                    'size': size,
                    'price': price,
                    'persistenceType': 'LAPSE'
                }
            }]
        }
        
        response = self._make_request('placeOrders', data=data)
        
        if response.get('status') == 'SUCCESS':
            bet_id = response['instructionReports'][0]['betId']
            order.bet_id = bet_id
            order.status = BetStatus.MATCHED
        
        return order
    
    def get_bet_status(self, bet_id: str) -> BetStatus:
        """Get current status of a bet."""
        if self.simulation_mode:
            order = self.simulated_bets.get(bet_id)
            return order.status if order else BetStatus.CANCELLED
        
        # Real API call
        data = {'betIds': [bet_id]}
        response = self._make_request('listCurrentOrders', data=data)
        
        if response.get('currentOrders'):
            status_str = response['currentOrders'][0]['status']
            return BetStatus(status_str)
        
        return BetStatus.CANCELLED
    
    def cancel_bet(self, bet_id: str) -> bool:
        """Cancel a bet."""
        if self.simulation_mode:
            if bet_id in self.simulated_bets:
                self.simulated_bets[bet_id].status = BetStatus.CANCELLED
                print(f"✓ [SIMULATION] Bet cancelled: {bet_id}")
                return True
            return False
        
        # Real API call
        data = {
            'marketId': None,  # Will be filled from bet details
            'instructions': [{'betId': bet_id}]
        }
        
        response = self._make_request('cancelOrders', data=data)
        return response.get('status') == 'SUCCESS'
    
    def get_market_odds(self, market_id: str) -> Dict:
        """Get current odds for a market."""
        if self.simulation_mode:
            # Return simulated odds
            return {
                'market_id': market_id,
                'runners': [
                    {'selection_id': '1', 'back_price': 2.5, 'lay_price': 2.52},
                    {'selection_id': '2', 'back_price': 3.2, 'lay_price': 3.25},
                    {'selection_id': '3', 'back_price': 2.8, 'lay_price': 2.84}
                ]
            }
        
        # Real API call
        data = {
            'marketIds': [market_id],
            'priceProjection': {
                'priceData': ['EX_BEST_OFFERS']
            }
        }
        
        response = self._make_request('listMarketBook', data=data)
        return response
