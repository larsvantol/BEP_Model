from math import log

from model.channel_properties import ChannelProperties


def characteristic_intrusion_length(properties: ChannelProperties, percentage=0.01) -> float:
    """
    Calculate the characteristic intrusion length for a channel.
    """
    return (
        (properties.width * properties.height * properties.effective_diffusion_coefficient)
        / (-abs(properties.tidal_averaged_flow))
    ) * log(percentage)
