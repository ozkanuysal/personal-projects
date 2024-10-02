from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
import sqlmodel
from sqlalchemy.dialects import postgresql

revision: str = 'a04d79012711'
down_revision: Union[str, None] = 'dba4f311e944'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('tags',
    sa.Column('uid', sa.UUID(), nullable=False),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(), nullable=True),
    sa.PrimaryKeyConstraint('uid')
    )
    op.create_table('booktag',
    sa.Column('book_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('tag_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.ForeignKeyConstraint(['book_id'], ['books.uid'], ),
    sa.ForeignKeyConstraint(['tag_id'], ['tags.uid'], ),
    sa.PrimaryKeyConstraint('book_id', 'tag_id')
    )


def downgrade() -> None:
    op.drop_table('booktag')
    op.drop_table('tags')
