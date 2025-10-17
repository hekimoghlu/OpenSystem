/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
#include "Dualshock3HIDGamepad.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(GAMEPAD) && PLATFORM(MAC)

#include "GamepadConstants.h"
#include "GamepadConstantsMac.h"
#include "Logging.h"
#include <IOKit/hid/IOHIDUsageTables.h>
#include <wtf/HexNumber.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Dualshock3HIDGamepad);

Dualshock3HIDGamepad::Dualshock3HIDGamepad(HIDDevice&& device, unsigned index)
    : HIDGamepad(WTFMove(device), index)
{
    LOG(Gamepad, "Creating Dualshock3HIDGamepad %p", this);

    m_mapping = standardGamepadMappingString();

    m_buttonValues = Vector(numberOfStandardGamepadButtonsWithHomeButton, SharedGamepadValue { 0.0 });

    constexpr size_t axisCount = 4;
    m_axisValues = Vector(axisCount, SharedGamepadValue { 0.0 });

    auto inputElements = hidDevice().uniqueInputElementsInDeviceTreeOrder();

    Vector<HIDElement> pointerElements;

    auto mapButton = [this] (HIDElement& element, GamepadButtonRole role) {
        m_elementMap.set(element.cookie(), makeUnique<HIDGamepadButton>(element, m_buttonValues[(size_t)role]));
    };

    // Look specifically for the axes and digital buttons
    for (auto& element : inputElements) {
        switch (element.fullUsage()) {
        case hidPointerFullUsage:
            pointerElements.append(element);
            break;
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
            mapButton(element, GamepadButtonRole::CenterClusterLeft);
            break;
        case hidButton2FullUsage:
            mapButton(element, GamepadButtonRole::LeftStick);
            break;
        case hidButton3FullUsage:
            mapButton(element, GamepadButtonRole::RightStick);
            break;
        case hidButton4FullUsage:
            mapButton(element, GamepadButtonRole::CenterClusterRight);
            break;
        case hidButton17FullUsage:
            mapButton(element, GamepadButtonRole::CenterClusterCenter);
            break;
        default:
            break;
        }
    }

    const size_t expectedGenericPointerElements = 40;
    if (pointerElements.size() != expectedGenericPointerElements) {
        LOG(Gamepad, "Dualshock3 controller was expected to have %lu generic pointer elements, has %lu instead", expectedGenericPointerElements, pointerElements.size());
        return;
    }

    mapButton(pointerElements[5], GamepadButtonRole::LeftClusterTop);
    mapButton(pointerElements[6], GamepadButtonRole::LeftClusterRight);
    mapButton(pointerElements[7], GamepadButtonRole::LeftClusterBottom);
    mapButton(pointerElements[8], GamepadButtonRole::LeftClusterLeft);
    mapButton(pointerElements[9], GamepadButtonRole::LeftShoulderBack);
    mapButton(pointerElements[10], GamepadButtonRole::RightShoulderBack);
    mapButton(pointerElements[11], GamepadButtonRole::LeftShoulderFront);
    mapButton(pointerElements[12], GamepadButtonRole::RightShoulderFront);
    mapButton(pointerElements[13], GamepadButtonRole::RightClusterTop);
    mapButton(pointerElements[14], GamepadButtonRole::RightClusterRight);
    mapButton(pointerElements[15], GamepadButtonRole::RightClusterBottom);
    mapButton(pointerElements[16], GamepadButtonRole::RightClusterLeft);
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && PLATFORM(MAC)
