class MaxLengthExceededError(Exception):
    """Exception raised when input exceeds the maximum allowed length."""
    def __init__(self, message="The input exceeds the maximum allowed length."):
        self.message = message
        super().__init__(self.message)