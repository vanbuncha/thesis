# db.py
import os
import asyncpg
import asyncio

_pool: asyncpg.pool.Pool | None = None


async def init_db_pool(retries: int = 10, delay: float = 1.0):
    global _pool
    for attempt in range(1, retries + 1):
        try:
            _pool = await asyncpg.create_pool(
                host=os.getenv("POSTGRES_HOST"),
                port=os.getenv("POSTGRES_PORT"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                database=os.getenv("POSTGRES_DB"),
            )
            print("✅ Connected to database")

            await _pool.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                  id SERIAL PRIMARY KEY,
                  room_id TEXT    NOT NULL,
                  sender  TEXT    NOT NULL,
                  role    TEXT    NOT NULL CHECK (role IN ('user','assistant')),
                  text    TEXT    NOT NULL,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            print("✅ Messages table exists")
            return

        except (asyncpg.exceptions.CannotConnectNowError, ConnectionError) as e:
            print(f"Database not ready (attempt {attempt}/{retries}): {e!r}")
            await asyncio.sleep(delay)

    raise RuntimeError("❌ Could not connect to database after retries")


async def save_message(room_id: str, sender: str, role: str, text: str):
    if _pool is None:
        raise RuntimeError("Database pool not initialized")
    await _pool.execute(
        """
        INSERT INTO messages(room_id, sender, role, text)
        VALUES($1, $2, $3, $4)
        """,
        room_id,
        sender,
        role,
        text,
    )
