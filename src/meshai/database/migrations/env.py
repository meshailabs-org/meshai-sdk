"""
Alembic Migration Environment for MeshAI
"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from meshai.database.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_url():
    """Get database URL from environment or config"""
    url = os.getenv("MESHAI_DATABASE_URL")
    if url:
        return url
    
    # Check for PostgreSQL config
    pg_host = os.getenv("MESHAI_POSTGRES_HOST")
    if pg_host:
        pg_user = os.getenv("MESHAI_POSTGRES_USER", "meshai")
        pg_password = os.getenv("MESHAI_POSTGRES_PASSWORD", "meshai")
        pg_database = os.getenv("MESHAI_POSTGRES_DATABASE", "meshai")
        pg_port = os.getenv("MESHAI_POSTGRES_PORT", "5432")
        return f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
    
    # Default to SQLite
    db_path = os.getenv("MESHAI_SQLITE_PATH", "meshai.db")
    return f"sqlite:///{db_path}"


def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
