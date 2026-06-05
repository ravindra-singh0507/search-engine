"""Phase 7 Workflow Templates."""
from app.workflows.templates import (
    WorkflowTemplate, ComparisonWorkflow, InvestigationWorkflow,
    DocumentationWorkflow, SummarizationWorkflow,
    TechEvalWorkflow, RootCauseWorkflow, get_workflow_registry,
)

__all__ = [
    "WorkflowTemplate", "ComparisonWorkflow", "InvestigationWorkflow",
    "DocumentationWorkflow", "SummarizationWorkflow",
    "TechEvalWorkflow", "RootCauseWorkflow", "get_workflow_registry",
]
