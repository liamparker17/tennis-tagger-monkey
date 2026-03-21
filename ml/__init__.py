"""Tennis Tagger ML Core — Pure inference modules."""
import logging
import torch

torch.set_num_threads(1)
logging.basicConfig(format="%(levelname)s %(name)s %(message)s", level=logging.INFO)
