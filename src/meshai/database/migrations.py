"""
Database Migrations for MeshAI

Handles database initialization, schema creation, and migrations.
"""

import os
from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
import structlog

from .models import Base
from .session import DatabaseManager

logger = structlog.get_logger(__name__)


def get_alembic_config(database_url: Optional[str] = None) -> Config:
    """Get Alembic configuration"""
    
    # Path to alembic directory
    migrations_dir = Path(__file__).parent / "migrations"
    alembic_cfg_path = migrations_dir / "alembic.ini"
    
    # Create migrations directory if it doesn't exist
    migrations_dir.mkdir(exist_ok=True)
    
    # Create alembic.ini if it doesn't exist
    if not alembic_cfg_path.exists():
        _create_alembic_ini(alembic_cfg_path, migrations_dir)
    
    # Load configuration
    alembic_cfg = Config(str(alembic_cfg_path))
    alembic_cfg.set_main_option("script_location", str(migrations_dir))
    
    # Set database URL if provided
    if database_url:
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    
    return alembic_cfg


def _create_alembic_ini(config_path: Path, migrations_dir: Path):
    """Create alembic.ini configuration file"""
    
    config_content = f"""# MeshAI Database Migrations Configuration

[alembic]
# path to migration scripts
script_location = {migrations_dir}

# template used to generate migration files
# file_template = %%(rev)s_%%(slug)s

# sys.path path, will be prepended to sys.path if present.
# defaults to the current working directory.
prepend_sys_path = .

# timezone to use when rendering the date within the migration file
# as well as the filename.
# If specified, requires the python-dateutil library that can be
# installed by adding `alembic[tz]` to the pip requirements
# string value is passed to dateutil.tz.gettz()
# leave blank for localtime
# timezone =

# max length of characters to apply to the
# "slug" field
# truncate_slug_length = 40

# set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
# revision_environment = false

# set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
# sourceless = false

# version path separator; As mentioned above, this is the character used to split
# version_locations. The default within new alembic.ini files is "os", which uses
# os.pathsep. If this key is omitted entirely, it falls back to the legacy
# behavior of splitting on spaces and/or commas.
# Valid values for version_path_separator are:
#
# version_path_separator = :
# version_path_separator = ;
# version_path_separator = space
version_path_separator = os

# the output encoding used when revision files
# are written from script.py.mako
# output_encoding = utf-8

sqlalchemy.url = sqlite:///meshai.db

[post_write_hooks]
# post_write_hooks defines scripts or Python functions that are run
# on newly generated revision scripts.  See the documentation for further
# detail and examples

# format using "black" - use the console_scripts runner, against the "black" entrypoint
# hooks = black
# black.type = console_scripts
# black.entrypoint = black
# black.options = -l 79 REVISION_SCRIPT_FILENAME

# Logging configuration
[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    logger.info(f"Created Alembic configuration: {config_path}")


def _create_env_py(migrations_dir: Path):
    """Create env.py for Alembic migrations"""
    
    env_py_path = migrations_dir / "env.py"
    
    if env_py_path.exists():
        return
    
    env_py_content = '''"""
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
'''
    
    with open(env_py_path, 'w') as f:
        f.write(env_py_content)
    
    logger.info(f"Created Alembic env.py: {env_py_path}")


def init_database(database_url: Optional[str] = None):
    """Initialize database with Alembic migrations"""
    
    # Get database manager and URL
    db_manager = DatabaseManager()
    if not database_url:
        database_url = db_manager._get_database_url(async_mode=False)
    
    logger.info(f"Initializing database: {database_url}")
    
    try:
        # Create migrations directory structure
        migrations_dir = Path(__file__).parent / "migrations"
        versions_dir = migrations_dir / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Create alembic configuration
        alembic_cfg = get_alembic_config(database_url)
        
        # Create env.py if it doesn't exist
        _create_env_py(migrations_dir)
        
        # Check if this is a fresh database
        with db_manager.sync_engine.connect() as conn:
            migration_context = MigrationContext.configure(conn)
            current_rev = migration_context.get_current_revision()
        
        if current_rev is None:
            # Fresh database - create all tables
            logger.info("Creating fresh database schema...")
            
            # Create all tables
            Base.metadata.create_all(db_manager.sync_engine)
            
            # Stamp with alembic version
            command.stamp(alembic_cfg, "head")
            
            logger.info("Database schema created successfully")
        else:
            logger.info(f"Database already initialized (revision: {current_rev})")
    
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def create_migration(message: str, autogenerate: bool = True):
    """Create a new database migration"""
    
    db_manager = DatabaseManager()
    database_url = db_manager._get_database_url(async_mode=False)
    
    alembic_cfg = get_alembic_config(database_url)
    
    logger.info(f"Creating migration: {message}")
    
    try:
        command.revision(
            alembic_cfg,
            message=message,
            autogenerate=autogenerate
        )
        logger.info("Migration created successfully")
    except Exception as e:
        logger.error(f"Failed to create migration: {e}")
        raise


def upgrade_database(revision: str = "head"):
    """Upgrade database to specified revision"""
    
    db_manager = DatabaseManager()
    database_url = db_manager._get_database_url(async_mode=False)
    
    alembic_cfg = get_alembic_config(database_url)
    
    logger.info(f"Upgrading database to revision: {revision}")
    
    try:
        command.upgrade(alembic_cfg, revision)
        logger.info("Database upgrade completed successfully")
    except Exception as e:
        logger.error(f"Failed to upgrade database: {e}")
        raise


def downgrade_database(revision: str):
    """Downgrade database to specified revision"""
    
    db_manager = DatabaseManager()
    database_url = db_manager._get_database_url(async_mode=False)
    
    alembic_cfg = get_alembic_config(database_url)
    
    logger.info(f"Downgrading database to revision: {revision}")
    
    try:
        command.downgrade(alembic_cfg, revision)
        logger.info("Database downgrade completed successfully")
    except Exception as e:
        logger.error(f"Failed to downgrade database: {e}")
        raise


def get_current_revision() -> Optional[str]:
    """Get current database revision"""
    
    db_manager = DatabaseManager()
    
    try:
        with db_manager.sync_engine.connect() as conn:
            migration_context = MigrationContext.configure(conn)
            return migration_context.get_current_revision()
    except Exception as e:
        logger.error(f"Failed to get current revision: {e}")
        return None


def get_migration_history():
    """Get migration history"""
    
    db_manager = DatabaseManager()
    database_url = db_manager._get_database_url(async_mode=False)
    
    alembic_cfg = get_alembic_config(database_url)
    
    try:
        script = ScriptDirectory.from_config(alembic_cfg)
        
        with db_manager.sync_engine.connect() as conn:
            migration_context = MigrationContext.configure(conn)
            
            # Get all revisions
            revisions = []
            for revision in script.walk_revisions():
                is_current = revision.revision == migration_context.get_current_revision()
                revisions.append({
                    "revision": revision.revision,
                    "message": revision.doc,
                    "is_current": is_current
                })
            
            return revisions
    
    except Exception as e:
        logger.error(f"Failed to get migration history: {e}")
        return []