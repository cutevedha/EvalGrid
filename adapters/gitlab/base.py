# GitLab Adapter Base - Interface for creating GitLab issues from evaluation results
# Implement this to automatically file GitLab issues when evaluations detect regressions

class GitLabClient:
    """
    Abstract base class for GitLab integration.

    Subclass this and implement create_issue() to connect the framework
    to a real GitLab instance so failed evaluations become actionable issues.
    """

    def create_issue(self, title: str, description: str):
        """
        Create a new GitLab issue.

        Args:
            title: Issue title (e.g. "Evaluation regression: policy_safe dropped below 1.0")
            description: Issue body with full evaluation details

        Returns:
            Created issue data (implementation-specific structure)
        """
        raise NotImplementedError
