/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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
#include "GenericHIDGamepad.h"

#if ENABLE(GAMEPAD) && PLATFORM(MAC)

#include "Logging.h"
#include <IOKit/hid/IOHIDUsageTables.h>
#include <wtf/HexNumber.h>

namespace WebCore {

GenericHIDGamepad::GenericHIDGamepad(HIDDevice&& device, unsigned index)
    : HIDGamepad(WTFMove(device), index)
{
    LOG(Gamepad, "Creating GenericHIDGamepad %p", this);

    auto inputElements = hidDevice().uniqueInputElementsInDeviceTreeOrder();

    for (auto& element : inputElements) {
        switch (element.usagePage()) {
        case kHIDPage_GenericDesktop:
            maybeAddGenericDesktopElement(element);
            continue;
        case kHIDPage_Button:
            maybeAddButtonElement(element);
            continue;
        default:
            continue;
        }
    }
}

void GenericHIDGamepad::maybeAddGenericDesktopElement(HIDElement& element)
{
    switch (element.usage()) {
    case kHIDUsage_GD_X:
    case kHIDUsage_GD_Y:
    case kHIDUsage_GD_Z:
    case kHIDUsage_GD_Rx:
    case kHIDUsage_GD_Ry:
    case kHIDUsage_GD_Rz:
        m_axisValues.append(0.0);
        m_elementMap.set(element.cookie(), makeUnique<HIDGamepadAxis>(element, m_axisValues.last()));
        break;
    case kHIDUsage_GD_DPadUp:
    case kHIDUsage_GD_DPadDown:
    case kHIDUsage_GD_DPadRight:
    case kHIDUsage_GD_DPadLeft:
        m_buttonValues.append(0.0);
        m_elementMap.set(element.cookie(), makeUnique<HIDGamepadButton>(element, m_buttonValues.last()));
        break;
    default:
        break;
    }
}

void GenericHIDGamepad::maybeAddButtonElement(HIDElement& element)
{
    // If it's in the button page, we assume it's actually a button.
    m_buttonValues.append(0.0);
    m_elementMap.set(element.cookie(), makeUnique<HIDGamepadButton>(element, m_buttonValues.last()));
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && PLATFORM(MAC)
