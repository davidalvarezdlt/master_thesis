import skeltorch
from .data import ThesisData
from .runner import ThesisRunner

skeltorch.Skeltorch(ThesisData(), ThesisRunner()).run()
