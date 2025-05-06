#!/usr/bin/python

"""
Allocate OTA lab radios with associated compute nodes with gnuradio installed.  

The OTA lab has the following resources

* Four NI X310 SDRs

These are paired with either d430 or d740 compute nodes (profile parameter)
when allocated by this profile, connected by 10G Ethernet.

* Four i7 NUC comput nodes with USB3-attached radios
  - One NI B210 per node 
  - One Quectel RM500Q-GL 5G modem per node

See the following diagram for the lab layout: [OTA Lab Layout](https://gitlab.flux.utah.edu/powderrenewpublic/powder-deployment/-/raw/master/diagrams/ota-lab.png)

Be sure to select at least one node/radio, or nothing  will be allocated!

If you plan to transmit anything, you will need to declare the frequency range
you plan to use for transmission via the parameters in this profile.  If you
transmit without declaring frequencies, this will be detected, and your
experiment may be shut down.  Please make sure the frequencies you declare
are compatible with the radios you specify for allocation.

Instructions:

GNU Radio 3.8 and UHD 4.0.0 are installed on all compute nodes instantiated
by this profile.  GNU Radio Companion is also present, and can be run through
an X session that is forwarded through SSH connections to the nodes.

Please be *certain* to configure GNU Radio and UHD tools to operate within the 
frequency range you have declared for your experiment!

"""

import geni.portal as portal
import geni.rspec.pg as rspec
import geni.rspec.emulab.pnext as pn
import geni.rspec.emulab.spectrum as spectrum
import geni.rspec.igext as ig

x310_node_disk_image = \
        "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU18-64-GR38-PACK"
b210_node_disk_image = \
        "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU18-64-GR38-PACK"


def x310_node_pair(x310_radio_name, node_type):
    radio_link = request.Link("%s-link" % x310_radio_name)
    radio_link.bandwidth = 10*1000*1000

    node = request.RawPC("%s-node" % x310_radio_name)
    node.hardware_type = node_type
    node.disk_image = x310_node_disk_image

    node_radio_if = node.addInterface("usrp_if")
    node_radio_if.addAddress(rspec.IPv4Address("192.168.40.1",
                                               "255.255.255.0"))
    radio_link.addInterface(node_radio_if)

    radio = request.RawPC("%s-radio"% x310_radio_name)
    radio.component_id = x310_radio_name
    radio_link.addNode(radio)

def b210_nuc_pair(b210_node):
    b210_nuc_pair_node = request.RawPC(b210_node.node_id)
    b210_nuc_pair_node.component_id = b210_node.node_id
    b210_nuc_pair_node.disk_image = b210_node_disk_image

portal.context.defineParameter("x310_comp_nodetype",
                               "Type of the node paired with the X310 Radios",
                               portal.ParameterType.STRING, "d430", ("d430", "d740"))

lab_x310_names = [
    "ota-x310-1",
    "ota-x310-2",
    "ota-x310-3",
    "ota-x310-4",
]

portal.context.defineStructParameter("x310_radios", "OTA Lab X310 Radios", [],
                                     multiValue=True,
                                     itemDefaultValue=
                                     {},
                                     min=0, max=None,
                                     members=[
                                        portal.Parameter(
                                             "radio_name",
                                             "OTA Lab X310",
                                             portal.ParameterType.STRING,
                                             lab_x310_names[0],
                                             lab_x310_names)
                                     ])

ota_b210_names = [
    "ota-nuc1",
    "ota-nuc2",
    "ota-nuc3",
    "ota-nuc4",
]

portal.context.defineStructParameter("b210_nodes", "OTA Lab B210 Radios", [],
                                     multiValue=True,
                                     min=0, max=None,
                                     members=[
                                         portal.Parameter(
                                             "node_id",
                                             "OTA Lab B210",
                                             portal.ParameterType.STRING,
                                             ota_b210_names[0],
                                             ota_b210_names)
                                     ],
                                    )


portal.context.defineStructParameter(
    "freq_ranges", "Frequency Ranges To Transmit In", [],
    multiValue=True,
    min=0,
    multiValueTitle="Frequency ranges to be used for transmission.",
    members=[
        portal.Parameter(
            "freq_min",
            "Frequency Range Min",
            portal.ParameterType.BANDWIDTH,
            3550.0,
            longDescription="Values are rounded to the nearest kilohertz."
        ),
        portal.Parameter(
            "freq_max",
            "Frequency Range Max",
            portal.ParameterType.BANDWIDTH,
            3560.0,
            longDescription="Values are rounded to the nearest kilohertz."
        ),
    ])

params = portal.context.bindParameters()

request = portal.context.makeRequestRSpec()

for i, x310_radio in enumerate(params.x310_radios):
    x310_node_pair(x310_radio.radio_name, params.x310_comp_nodetype)

for b210_node in params.b210_nodes:
    b210_nuc_pair(b210_node)

# Request frequency range(s)
for frange in params.freq_ranges:
    request.requestSpectrum(frange.freq_min, frange.freq_max, 0)

portal.context.printRequestRSpec()