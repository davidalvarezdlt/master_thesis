import skeltorch
from thesis.data import ThesisData
from .runner import ThesisCPNRunner

# Create and run Skeltorch object
skeltorch.Skeltorch(ThesisData(), ThesisCPNRunner()).run()