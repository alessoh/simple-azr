"""
Task implementations for the AZR system.

This package contains implementations of the three core reasoning tasks:
- Abduction: Input prediction given function and output
- Deduction: Output prediction given function and input
- Induction: Function synthesis given input-output pairs
"""

from azr.tasks.abduction import AbductionTask
from azr.tasks.deduction import DeductionTask
from azr.tasks.induction import InductionTask

__all__ = ['AbductionTask', 'DeductionTask', 'InductionTask']