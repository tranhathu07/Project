"""add content

Revision ID: 30eba9b7c840
Revises: 7b28a75bc1cd
Create Date: 2026-09-05 12:11:44.818999

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '30eba9b7c840'
down_revision: Union[str, Sequence[str], None] = '7b28a75bc1cd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('posts', sa.Column('content', sa.String(), nullable= False))
    pass


def downgrade() -> None:
    op.drop_column('posts', 'content')
    pass
