from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from src.core.config import settings

from src.modules.users.models import User
from src.engines.guardrail.models import GuardrailLog
from src.modules.imagery.models import ImageJob

# Create async engine
engine = create_async_engine(settings.DATABASE_URL, echo=True, future=True)

# Create async session factory
async_session_maker = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

async def get_session() -> AsyncSession:
    async with async_session_maker() as session:
        yield session
