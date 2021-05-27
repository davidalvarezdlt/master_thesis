import skeltorch
from thesis.data import ThesisData
from .runner import ThesisAlignmentRunner

# Create and run Skeltorch object
skeltorch.Skeltorch(ThesisData(), ThesisAlignmentRunner()).run()
