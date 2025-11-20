"""
Custom exception hierarchy for EXMO gait analysis pipeline.

Provides clear, actionable error messages with context for debugging.
Enables fine-grained exception handling at different pipeline stages.

Version: 1.0.0
Date: 2025-11-21
"""

from typing import Optional, Dict, Any


# ============================================================================
# Base Exception
# ============================================================================

class ExmoGaitError(Exception):
    """
    Base exception for all EXMO gait analysis errors.

    All custom exceptions inherit from this to allow catching all
    pipeline-specific errors with a single except clause.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Args:
            message: Human-readable error description
            details: Optional dict with additional context (file paths, values, etc.)
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Format error with details if available"""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{details_str}]"
        return self.message


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(ExmoGaitError):
    """Base class for configuration-related errors"""
    pass


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when config file doesn't exist"""

    def __init__(self, config_path: str):
        super().__init__(
            f"Configuration file not found: {config_path}",
            details={"config_path": config_path}
        )


class ConfigValidationError(ConfigurationError):
    """Raised when config contains invalid values"""

    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Invalid configuration value for '{field}': {reason}",
            details={"field": field, "value": value, "reason": reason}
        )


class IncompatibleConfigError(ConfigurationError):
    """Raised when config contains incompatible parameter combinations"""

    def __init__(self, conflicting_params: Dict[str, Any], reason: str):
        super().__init__(
            f"Incompatible configuration: {reason}",
            details={"conflicting_params": conflicting_params, "reason": reason}
        )


# ============================================================================
# Data Loading Errors
# ============================================================================

class DataLoadError(ExmoGaitError):
    """Base class for data loading errors"""
    pass


class FileNotFoundError(DataLoadError):
    """Raised when input CSV file doesn't exist"""

    def __init__(self, file_path: str, view: str):
        super().__init__(
            f"{view.upper()} view CSV not found: {file_path}",
            details={"file_path": file_path, "view": view}
        )


class InvalidCSVFormatError(DataLoadError):
    """Raised when CSV doesn't match expected DeepLabCut format"""

    def __init__(self, file_path: str, reason: str):
        super().__init__(
            f"Invalid CSV format in {file_path}: {reason}",
            details={"file_path": file_path, "reason": reason}
        )


class MissingKeypointsError(DataLoadError):
    """Raised when required keypoints are missing from CSV"""

    def __init__(self, file_path: str, missing_keypoints: list, view: str):
        super().__init__(
            f"{view.upper()} view missing keypoints: {', '.join(missing_keypoints)}",
            details={
                "file_path": file_path,
                "view": view,
                "missing_keypoints": missing_keypoints
            }
        )


class InsufficientDataError(DataLoadError):
    """Raised when data doesn't meet minimum requirements"""

    def __init__(self, reason: str, required: Optional[int] = None, actual: Optional[int] = None):
        details = {"reason": reason}
        if required is not None:
            details["required"] = required
        if actual is not None:
            details["actual"] = actual

        super().__init__(
            f"Insufficient data: {reason}",
            details=details
        )


# ============================================================================
# Processing Errors
# ============================================================================

class ProcessingError(ExmoGaitError):
    """Base class for data processing errors"""
    pass


class ScalingError(ProcessingError):
    """Raised when spatial scaling computation fails"""

    def __init__(self, reason: str, frames_used: Optional[int] = None):
        super().__init__(
            f"Scaling computation failed: {reason}",
            details={"reason": reason, "frames_used": frames_used}
        )


class SmoothingError(ProcessingError):
    """Raised when trajectory smoothing fails"""

    def __init__(self, keypoint: str, reason: str):
        super().__init__(
            f"Smoothing failed for keypoint '{keypoint}': {reason}",
            details={"keypoint": keypoint, "reason": reason}
        )


class PhaseDetectionError(ProcessingError):
    """Raised when walking/stationary phase detection fails"""

    def __init__(self, reason: str):
        super().__init__(
            f"Phase detection failed: {reason}",
            details={"reason": reason}
        )


class StepDetectionError(ProcessingError):
    """Raised when foot strike detection fails"""

    def __init__(self, paw: str, reason: str):
        super().__init__(
            f"Step detection failed for {paw}: {reason}",
            details={"paw": paw, "reason": reason}
        )


# ============================================================================
# Metric Computation Errors
# ============================================================================

class MetricComputationError(ExmoGaitError):
    """Base class for metric calculation errors"""
    pass


class InvalidMetricValueError(MetricComputationError):
    """Raised when computed metric is outside valid range"""

    def __init__(self, metric_name: str, value: float, valid_range: tuple):
        super().__init__(
            f"Invalid value for '{metric_name}': {value} (expected {valid_range[0]}-{valid_range[1]})",
            details={
                "metric_name": metric_name,
                "value": value,
                "valid_range": valid_range
            }
        )


class ROMComputationError(MetricComputationError):
    """Raised when range of motion calculation fails"""

    def __init__(self, joint: str, reason: str):
        super().__init__(
            f"ROM computation failed for {joint}: {reason}",
            details={"joint": joint, "reason": reason}
        )


class GaitMetricError(MetricComputationError):
    """Raised when gait metric calculation fails"""

    def __init__(self, metric: str, reason: str):
        super().__init__(
            f"Gait metric '{metric}' computation failed: {reason}",
            details={"metric": metric, "reason": reason}
        )


# ============================================================================
# Export Errors
# ============================================================================

class ExportError(ExmoGaitError):
    """Base class for export/output errors"""
    pass


class ExcelExportError(ExportError):
    """Raised when Excel export fails"""

    def __init__(self, output_path: str, reason: str):
        super().__init__(
            f"Excel export failed: {reason}",
            details={"output_path": output_path, "reason": reason}
        )


class PlotGenerationError(ExportError):
    """Raised when plot generation fails"""

    def __init__(self, plot_type: str, reason: str):
        super().__init__(
            f"Plot generation failed for '{plot_type}': {reason}",
            details={"plot_type": plot_type, "reason": reason}
        )


class OutputDirectoryError(ExportError):
    """Raised when output directory creation/access fails"""

    def __init__(self, directory: str, reason: str):
        super().__init__(
            f"Output directory error: {reason}",
            details={"directory": directory, "reason": reason}
        )


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(ExmoGaitError):
    """Base class for validation errors"""
    pass


class DataQualityError(ValidationError):
    """Raised when data doesn't meet quality thresholds"""

    def __init__(self, quality_metric: str, value: float, threshold: float):
        super().__init__(
            f"Data quality below threshold: {quality_metric}={value:.2%} (required: {threshold:.2%})",
            details={
                "quality_metric": quality_metric,
                "value": value,
                "threshold": threshold
            }
        )


class BiomechanicalConstraintError(ValidationError):
    """Raised when results violate biomechanical constraints"""

    def __init__(self, constraint: str, value: float, valid_range: tuple):
        super().__init__(
            f"Biomechanical constraint violated: {constraint}={value} (valid: {valid_range[0]}-{valid_range[1]})",
            details={
                "constraint": constraint,
                "value": value,
                "valid_range": valid_range
            }
        )


# ============================================================================
# Pipeline Errors
# ============================================================================

class PipelineError(ExmoGaitError):
    """Base class for pipeline execution errors"""
    pass


class PipelineStageError(PipelineError):
    """Raised when a pipeline stage fails"""

    def __init__(self, stage_name: str, original_error: Exception):
        super().__init__(
            f"Pipeline stage '{stage_name}' failed: {str(original_error)}",
            details={
                "stage_name": stage_name,
                "original_error": type(original_error).__name__,
                "error_message": str(original_error)
            }
        )
        self.original_error = original_error


class PipelineStateError(PipelineError):
    """Raised when pipeline state is invalid"""

    def __init__(self, missing_data: list, stage: str):
        super().__init__(
            f"Invalid pipeline state at stage '{stage}': missing {', '.join(missing_data)}",
            details={"stage": stage, "missing_data": missing_data}
        )


# ============================================================================
# Utility Functions
# ============================================================================

def format_error_chain(error: Exception) -> str:
    """
    Format exception chain for logging.

    Args:
        error: Exception to format

    Returns:
        Multi-line string with full error chain
    """
    lines = [f"Error: {type(error).__name__}: {str(error)}"]

    if isinstance(error, ExmoGaitError) and error.details:
        lines.append("Details:")
        for key, value in error.details.items():
            lines.append(f"  {key}: {value}")

    if hasattr(error, '__cause__') and error.__cause__:
        lines.append("\nCaused by:")
        lines.append(format_error_chain(error.__cause__))

    return "\n".join(lines)


def raise_with_context(error_class: type, message: str, original_error: Exception = None, **details):
    """
    Raise exception with chained context.

    Args:
        error_class: Exception class to raise
        message: Error message
        original_error: Optional original exception to chain
        **details: Additional context details

    Raises:
        error_class instance with chained context
    """
    new_error = error_class(message, details=details)

    if original_error:
        raise new_error from original_error
    else:
        raise new_error
