from src.models import MistralHf
from src.models.modeling_args import MistralMoEArgsHf


class MistralMoEHf(MistralHf):
    def __init__(self, args: MistralMoEArgsHf):
        super().__init__(args)

