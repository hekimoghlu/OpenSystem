"""Add rating range check constraint

Revision ID: 4cea4a5fe287
Revises: c856bd600df0
Create Date: 2024-10-28 07:19:44.536333

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4cea4a5fe287'
down_revision = 'c856bd600df0'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands manually added ###
    op.create_check_constraint('review_rating_constraint', 'reviews', 'rating >=0 and rating <= 100')
    # ### end Alembic commands ###


def downgrade():
    # ### commands manually added ###
    op.drop_constraint('review_rating_constraint', 'reviews', 'check')
    # ### end Alembic commands ###
