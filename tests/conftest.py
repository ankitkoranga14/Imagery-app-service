import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app
from typing import AsyncGenerator

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

@pytest.fixture(scope="module")
async def client() -> AsyncGenerator[AsyncClient, None]:
    # Trigger lifespan events (startup/shutdown)
    # app.router.lifespan_context(app) returns an async context manager
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac
