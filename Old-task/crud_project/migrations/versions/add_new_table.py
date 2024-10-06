from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
import sqlmodel
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "dba4f311e944"
down_revision: Union[str, None] = "11d1f79aef4d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "reviews",
        sa.Column("uid", sa.UUID(), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("review_text", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("user_uid", sqlmodel.sql.sqltypes.GUID(), nullable=True),
        sa.Column("book_uid", sqlmodel.sql.sqltypes.GUID(), nullable=True),
        sa.Column("created_at", postgresql.TIMESTAMP(), nullable=True),
        sa.Column("update_at", postgresql.TIMESTAMP(), nullable=True),
        sa.ForeignKeyConstraint(
            ["book_uid"],
            ["books.uid"],
        ),
        sa.ForeignKeyConstraint(
            ["user_uid"],
            ["users.uid"],
        ),
        sa.PrimaryKeyConstraint("uid"),
    )


def downgrade() -> None:
    op.drop_table("reviews")    