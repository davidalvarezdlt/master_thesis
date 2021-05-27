import skeltorch
from thesis.data import ThesisData
from .runner import ThesisInpaintingRunner

# Create and run Skeltorch object
skeltorch.Skeltorch(ThesisData(), ThesisInpaintingRunner()).run()
