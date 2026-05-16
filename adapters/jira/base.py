# Jira Adapter Base - Interface for creating and managing Jira issues
# Implement this to push failed evaluation results as Jira tickets automatically

class JiraClient:
    """
    Abstract base class for Jira integration.

    Subclass this and implement get_issue() to connect the framework
    to a real Jira instance so failed evaluations can be auto-triaged.
    """

    def get_issue(self, issue_id: str):
        """
        Retrieve a Jira issue by its ID.

        Args:
            issue_id: Jira issue identifier (e.g. "PROJ-123")

        Returns:
            Issue data dict (implementation-specific structure)
        """
        raise NotImplementedError
