
def set_UHD_USRP_GPIO_TX(uhd_usrp_sink):
    uhd_usrp_sink.set_gpio_attr("FP0", "DDR", 0x10, 0x10, 0)
    uhd_usrp_sink.set_gpio_attr("FP0", "OUT", 0x10, 0x10, 0)
    
def set_UHD_USRP_GPIO_RX(uhd_usrp_source):
    uhd_usrp_source.set_gpio_attr("FP0", "CTRL", 0x00)
    uhd_usrp_source.set_gpio_attr("FP0", "DDR",  0x10, 0x10, 0)
    uhd_usrp_source.set_gpio_attr("FP0", "OUT",  0x10, 0x10, 0)
    