class DigitRecognizerError(Exception):
    """Base class for expected digit-recognizer errors."""


class InvalidImageError(DigitRecognizerError):
    """Raised when an image input is unsupported or malformed."""


class BlankImageError(DigitRecognizerError):
    """Raised when an image contains no meaningful ink."""


class ModelLoadError(DigitRecognizerError):
    """Raised when the digit-recognition model cannot be loaded."""
