from robosuite_ohio.environments.base import make

# Manipulation environments
from robosuite_ohio.environments.manipulation.lift import Lift
from robosuite_ohio.environments.manipulation.stack import Stack
from robosuite_ohio.environments.manipulation.nut_assembly import NutAssembly
from robosuite_ohio.environments.manipulation.pick_place import PickPlace
from robosuite_ohio.environments.manipulation.door import Door
from robosuite_ohio.environments.manipulation.wipe import Wipe
from robosuite_ohio.environments.manipulation.tool_hang import ToolHang
from robosuite_ohio.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite_ohio.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole
from robosuite_ohio.environments.manipulation.two_arm_handover import TwoArmHandover
from robosuite_ohio.environments.manipulation.two_arm_transport import TwoArmTransport

from robosuite_ohio.environments import ALL_ENVIRONMENTS
from robosuite_ohio.controllers import ALL_CONTROLLERS, load_controller_config
from robosuite_ohio.robots import ALL_ROBOTS
from robosuite_ohio.models.grippers import ALL_GRIPPERS

__version__ = "1.4.1"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
