/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "config.h"
#include "LogitechGamepad.h"

#if ENABLE(GAMEPAD) && PLATFORM(MAC)

#include "GamepadConstants.h"
#include "GamepadConstantsMac.h"
#include "Logging.h"
#include <IOKit/hid/IOHIDUsageTables.h>
#include <wtf/HexNumber.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LogitechGamepad);

LogitechGamepad::LogitechGamepad(HIDDevice&& device, unsigned index)
    : HIDGamepad(WTFMove(device), index)
{
    LOG(Gamepad, "Creating LogitechGamepad %p", this);

    m_mapping = standardGamepadMappingString();

    m_buttonValues = Vector(numberOfStandardGamepadButtonsWithoutHomeButton, SharedGamepadValue { 0.0 });

    constexpr size_t axisCount = 4;
    m_axisValues = Vector(axisCount, SharedGamepadValue { 0.0 });

    auto inputElements = hidDevice().uniqueInputElementsInDeviceTreeOrder();

    auto mapButton = [this] (HIDElement& element, GamepadButtonRole role) {
        m_elementMap.set(element.cookie(), makeUnique<HIDGamepadButton>(element, m_buttonValues[(size_t)role]));
    };

    for (auto& element : inputElements) {
        switch (element.fullUsage()) {
        case hidHatswitchFullUsage: {
            auto hatswitchValues = Vector {
                m_buttonValues[(size_t)GamepadButtonRole::LeftClusterTop],
                m_buttonValues[(size_t)GamepadButtonRole::LeftClusterRight],
                m_buttonValues[(size_t)GamepadButtonRole::LeftClusterBottom],
                m_buttonValues[(size_t)GamepadButtonRole::LeftClusterLeft]
            };

            m_elementMap.set(element.cookie(), makeUnique<HIDGamepadHatswitch>(element, WTFMove(hatswitchValues)));
            break;
        }
        case hidXAxisFullUsage:
            m_elementMap.set(element.cookie(), makeUnique<HIDGamepadAxis>(element, m_axisValues[0]));
            break;
        case hidYAxisFullUsage:
            m_elementMap.set(element.cookie(), makeUnique<HIDGamepadAxis>(element, m_axisValues[1]));
            break;
        case hidZAxisFullUsage:
            m_elementMap.set(element.cookie(), makeUnique<HIDGamepadAxis>(element, m_axisValues[2]));
            break;
        case hidRzAxisFullUsage:
            m_elementMap.set(element.cookie(), makeUnique<HIDGamepadAxis>(element, m_axisValues[3]));
            break;
        case hidButton1FullUsage:
            mapButton(element, GamepadButtonRole::RightClusterLeft);
            break;
        case hidButton2FullUsage:
            mapButton(element, GamepadButtonRole::RightClusterBottom);
            break;
        case hidButton3FullUsage:
            mapButton(element, GamepadButtonRole::RightClusterRight);
            break;
        case hidButton4FullUsage:
            mapButton(element, GamepadButtonRole::RightClusterTop);
            break;
        case hidButton5FullUsage:
            mapButton(element, GamepadButtonRole::LeftShoulderFront);
            break;
        case hidButton6FullUsage:
            mapButton(element, GamepadButtonRole::RightShoulderFront);
            break;
        case hidButton7FullUsage:
            mapButton(element, GamepadButtonRole::LeftShoulderBack);
            break;
        case hidButton8FullUsage:
            mapButton(element, GamepadButtonRole::RightShoulderBack);
            break;
        case hidButton9FullUsage:
            mapButton(element, GamepadButtonRole::CenterClusterLeft);
            break;
        case hidButton10FullUsage:
            mapButton(element, GamepadButtonRole::CenterClusterRight);
            break;
        case hidButton11FullUsage:
            mapButton(element, GamepadButtonRole::LeftStick);
            break;
        case hidButton12FullUsage:
            mapButton(element, GamepadButtonRole::RightStick);
            break;
        default:
            break;
        }
    }
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && PLATFORM(MAC)
