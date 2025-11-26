"""
Environments package for RAGEN
"""
from .base import MultiTurnEnvironment

# Import environments conditionally to avoid requiring WebShop dependencies
try:
    from .webshop import WebShopEnvironment
except ImportError:
    WebShopEnvironment = None

from .medium_webshop import MediumWebShopEnvironment

__all__ = ['MultiTurnEnvironment', 'MediumWebShopEnvironment']
if WebShopEnvironment is not None:
    __all__.append('WebShopEnvironment')