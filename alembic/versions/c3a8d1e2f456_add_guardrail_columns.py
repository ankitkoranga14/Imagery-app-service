"""Add missing columns to guardrail_logs and new tables

Revision ID: c3a8d1e2f456
Revises: ba39b2c7cd93
Create Date: 2026-01-13 08:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = 'c3a8d1e2f456'
down_revision: Union[str, Sequence[str], None] = 'ba39b2c7cd93'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add missing columns to guardrail_logs and create new tables."""
    
    # Add missing columns to guardrail_logs table
    # Using batch mode for SQLite compatibility
    with op.batch_alter_table('guardrail_logs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('input_hash', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('levels_executed', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('levels_failed', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('darkness_score', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('glare_percentage', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('food_score', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('nsfw_score', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('dish_count', sa.Integer(), nullable=True))
        batch_op.create_index('ix_guardrail_logs_input_hash', ['input_hash'], unique=False)
    
    # Create guardrail_feedback table if it doesn't exist
    op.create_table(
        'guardrail_feedback',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('validation_id', sa.Integer(), nullable=True),
        sa.Column('input_hash', sa.String(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('feedback_type', sa.String(), nullable=False),
        sa.Column('failed_level', sa.String(), nullable=True),
        sa.Column('user_comment', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('original_prompt', sa.String(), nullable=True),
        sa.Column('original_status', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['validation_id'], ['guardrail_logs.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_guardrail_feedback_input_hash', 'guardrail_feedback', ['input_hash'], unique=False)
    
    # Create guardrail_config_variants table if it doesn't exist
    op.create_table(
        'guardrail_config_variants',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('config_json', sa.String(), nullable=False),
        sa.Column('rollout_percentage', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('total_validations', sa.Integer(), nullable=False),
        sa.Column('false_positives', sa.Integer(), nullable=False),
        sa.Column('false_negatives', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_guardrail_config_variants_name', 'guardrail_config_variants', ['name'], unique=True)


def downgrade() -> None:
    """Remove added columns and tables."""
    
    # Drop new tables
    op.drop_index('ix_guardrail_config_variants_name', table_name='guardrail_config_variants')
    op.drop_table('guardrail_config_variants')
    
    op.drop_index('ix_guardrail_feedback_input_hash', table_name='guardrail_feedback')
    op.drop_table('guardrail_feedback')
    
    # Remove columns from guardrail_logs
    with op.batch_alter_table('guardrail_logs', schema=None) as batch_op:
        batch_op.drop_index('ix_guardrail_logs_input_hash')
        batch_op.drop_column('dish_count')
        batch_op.drop_column('nsfw_score')
        batch_op.drop_column('food_score')
        batch_op.drop_column('glare_percentage')
        batch_op.drop_column('darkness_score')
        batch_op.drop_column('levels_failed')
        batch_op.drop_column('levels_executed')
        batch_op.drop_column('input_hash')

