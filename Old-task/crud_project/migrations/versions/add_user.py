
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
from sqlalchemy.dialects import postgresql

revision: str = "11d1f79aef4d"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("uid", sa.UUID(), nullable=False),
        sa.Column("username", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("email", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("first_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("last_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("role", sa.VARCHAR(), server_default="user", nullable=False),
        sa.Column("is_verified", sa.Boolean(), nullable=False),
        sa.Column("password_hash", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(), nullable=True),
        sa.Column("update_at", postgresql.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint("uid"),
    )
    op.create_table(
        "books",
        sa.Column("uid", sa.UUID(), nullable=False),
        sa.Column("title", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("author", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("publisher", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("published_date", sa.Date(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("language", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("user_uid", sqlmodel.sql.sqltypes.GUID(), nullable=True),
        sa.Column("created_at", postgresql.TIMESTAMP(), nullable=True),
        sa.Column("update_at", postgresql.TIMESTAMP(), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_uid"],
            ["users.uid"],
        ),
        sa.PrimaryKeyConstraint("uid"),
    )


def downgrade() -> None:
    op.drop_table("books")
    op.drop_table("users")
