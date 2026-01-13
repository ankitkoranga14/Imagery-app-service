from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from src.core.config import settings

# Import ALL models to ensure they're registered with SQLModel.metadata
from src.modules.users.models import User
from src.engines.guardrail.models import GuardrailLog, GuardrailFeedback, GuardrailConfigVariant
from src.modules.imagery.models import ImageJob

# Create async engine
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)

# Create async session factory
async_session_maker = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def create_db_and_tables():
    """Create all tables if they don't exist.
    
    Uses checkfirst=True to avoid errors when tables already exist.
    """
    async with engine.begin() as conn:
        # Use checkfirst=True to avoid "table already exists" errors
        await conn.run_sync(lambda sync_conn: SQLModel.metadata.create_all(sync_conn, checkfirst=True))

async def get_session() -> AsyncSession:
    async with async_session_maker() as session:
        yield session
