# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from .ant import Ant
from .anymal import Anymal
from .anymal_terrain import AnymalTerrain
from .ball_balance import BallBalance
from .cartpole import Cartpole 
from .factory.factory_task_gears import FactoryTaskGears
from .factory.factory_task_insertion import FactoryTaskInsertion
from .factory.factory_task_nut_bolt_pick import FactoryTaskNutBoltPick
from .factory.factory_task_nut_bolt_place import FactoryTaskNutBoltPlace
from .factory.factory_task_nut_bolt_screw import FactoryTaskNutBoltScrew
from .franka_cabinet import FrankaCabinet
from .franka_cube_stack import FrankaCubeStack
from .humanoid import Humanoid
from .humanoid_amp import HumanoidAMP
from .ingenuity import Ingenuity
from .quadcopter import Quadcopter
from .shadow_hand import ShadowHand
from .shadow_hand_spin import ShadowHandSpin
from .shadow_hand_upside_down import ShadowHandUpsideDown
from .allegro_hand import AllegroHand
from .dextreme.allegro_hand_dextreme import AllegroHandDextremeManualDR, AllegroHandDextremeADR
from .trifinger import Trifinger

from .shadow_hand_block_stack import ShadowHandBlockStack
from .shadow_hand_bottle_cap import ShadowHandBottleCap
from .shadow_hand_catch_abreast import ShadowHandCatchAbreast
from .shadow_hand_catch_over2underarm import ShadowHandCatchOver2Underarm
from .shadow_hand_catch_underarm import ShadowHandCatchUnderarm
from .shadow_hand_door_close_inward import ShadowHandDoorCloseInward
from .shadow_hand_door_close_outward import ShadowHandDoorCloseOutward
from .shadow_hand_door_open_inward import ShadowHandDoorOpenInward
from .shadow_hand_door_open_outward import ShadowHandDoorOpenOutward
from .shadow_hand_grasp_and_place import ShadowHandGraspAndPlace
from .shadow_hand_kettle import ShadowHandKettle
from .shadow_hand_lift_underarm import ShadowHandLiftUnderarm
from .shadow_hand_over import ShadowHandOver
from .shadow_hand_pen import ShadowHandPen
from .shadow_hand_push_block import ShadowHandPushBlock
from .shadow_hand_re_orientation import ShadowHandReOrientation
from .shadow_hand_scissors import ShadowHandScissors
from .shadow_hand_swing_cup import ShadowHandSwingCup
from .shadow_hand_switch import ShadowHandSwitch
from .shadow_hand_two_catch_underarm import ShadowHandTwoCatchUnderarm

# Mappings from strings to environments
isaacgym_task_map = {
    "AllegroHand": AllegroHand,
    "AllegroHandManualDR": AllegroHandDextremeManualDR,
    "AllegroHandADR": AllegroHandDextremeADR,
    "Ant": Ant,
    "Anymal": Anymal,
    "AnymalTerrain": AnymalTerrain,
    "BallBalance": BallBalance,
    "Cartpole": Cartpole,
    "FactoryTaskGears": FactoryTaskGears,
    "FactoryTaskInsertion": FactoryTaskInsertion,
    "FactoryTaskNutBoltPick": FactoryTaskNutBoltPick,
    "FactoryTaskNutBoltPlace": FactoryTaskNutBoltPlace,
    "FactoryTaskNutBoltScrew": FactoryTaskNutBoltScrew,
    "FrankaCabinet": FrankaCabinet,
    "FrankaCubeStack": FrankaCubeStack,
    "Humanoid": Humanoid,
    "HumanoidAMP": HumanoidAMP,
    "Ingenuity": Ingenuity,
    "Quadcopter": Quadcopter,
    "ShadowHand": ShadowHand,
    "ShadowHandSpin": ShadowHandSpin, 
    "ShadowHandUpsideDown": ShadowHandUpsideDown,
    "Trifinger": Trifinger,

    "ShadowHandBlockStack": ShadowHandBlockStack,
    "ShadowHandBottleCap": ShadowHandBottleCap,
    "ShadowHandCatchAbreast": ShadowHandCatchAbreast,
    "ShadowHandCatchOver2Underarm": ShadowHandCatchOver2Underarm,
    "ShadowHandCatchUnderarm": ShadowHandCatchUnderarm,
    "ShadowHandDoorCloseInward": ShadowHandDoorCloseInward,
    "ShadowHandDoorCloseOutward": ShadowHandDoorCloseOutward,
    "ShadowHandDoorOpenInward": ShadowHandDoorOpenInward,
    "ShadowHandDoorOpenOutward": ShadowHandDoorOpenOutward,
    "ShadowHandGraspAndPlace": ShadowHandGraspAndPlace,
    "ShadowHandKettle": ShadowHandKettle,
    "ShadowHandLiftUnderarm": ShadowHandLiftUnderarm,
    "ShadowHandOver": ShadowHandOver,
    "ShadowHandPen": ShadowHandPen,
    "ShadowHandPushBlock": ShadowHandPushBlock,
    "ShadowHandReOrientation": ShadowHandReOrientation,
    "ShadowHandScissors": ShadowHandScissors,
    "ShadowHandSwingCup": ShadowHandSwingCup,
    "ShadowHandSwitch": ShadowHandSwitch,
    "ShadowHandTwoCatchUnderarm": ShadowHandTwoCatchUnderarm,
}

