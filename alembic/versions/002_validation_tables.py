"""Add validation_logs and validation_jobs tables

Revision ID: 002_validation_tables
Revises: 
Create Date: 2026-01-13

This migration adds the tables required for the Guardrail Microservice:
- validation_logs: Audit trail for all validation checks
- validation_jobs: Async job tracking for background validation
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '002_validation_tables'
down_revision = 'c3a8d1e2f456'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create validation_logs table
    op.create_table(
        'validation_logs',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('image_hash', sa.String(), nullable=False, index=True),
        sa.Column('image_url', sa.String(), nullable=True),
        sa.Column('prompt', sa.Text(), nullable=False, server_default=''),
        sa.Column('status', sa.String(10), nullable=False, server_default='PASS'),
        sa.Column('failure_reason', sa.Text(), nullable=True),
        sa.Column('failure_level', sa.String(20), nullable=True),
        sa.Column('scores', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('latency_ms', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('latency_breakdown', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('cache_hit', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('parallel_execution', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('levels_executed', sa.String(100), nullable=False, server_default=''),
        sa.Column('clip_variant', sa.String(50), nullable=True),
        sa.Column('yolo_variant', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'), index=True),
    )
    
    # Create validation_jobs table
    op.create_table(
        'validation_jobs',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='PENDING'),
        sa.Column('prompt', sa.Text(), nullable=False, server_default=''),
        sa.Column('image_hash', sa.String(), nullable=False, server_default=''),
        sa.Column('image_storage_key', sa.String(), nullable=True),
        sa.Column('validation_log_id', sa.String(), sa.ForeignKey('validation_logs.id'), nullable=True),
        sa.Column('result_status', sa.String(10), nullable=True),
        sa.Column('result_reason', sa.Text(), nullable=True),
        sa.Column('result_scores', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('result_latency_ms', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('celery_task_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'), index=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
    )
    
    # Create indexes for common queries
    op.create_index('ix_validation_logs_status', 'validation_logs', ['status'])
    op.create_index('ix_validation_logs_cache_hit', 'validation_logs', ['cache_hit'])
    op.create_index('ix_validation_jobs_status', 'validation_jobs', ['status'])


def downgrade() -> None:
    op.drop_index('ix_validation_jobs_status', 'validation_jobs')
    op.drop_index('ix_validation_logs_cache_hit', 'validation_logs')
    op.drop_index('ix_validation_logs_status', 'validation_logs')
    op.drop_table('validation_jobs')
    op.drop_table('validation_logs')
