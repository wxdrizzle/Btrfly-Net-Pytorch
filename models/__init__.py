from .btrfly_net import BtrflyNet
from .eb_discriminator import EBGAN


def build_model(cfg, name="Btrfly"):
    if name == "EBGAN":
        return EBGAN()
    return BtrflyNet(cfg)