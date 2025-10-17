/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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
#include "HIDGamepadElement.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(GAMEPAD) && PLATFORM(MAC)

#include <IOKit/hid/IOHIDElement.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(HIDGamepadElement);

#pragma mark HIDGamepadElement

HIDGamepadElement::HIDGamepadElement(const HIDElement& element)
    : HIDElement(element)
{
}

void HIDGamepadElement::refreshCurrentValue()
{
    IOHIDValueRef value;
    if (IOHIDDeviceGetValue(IOHIDElementGetDevice(rawElement()), rawElement(), &value) == kIOReturnSuccess)
        gamepadValueChanged(value);
}

double HIDGamepadElement::normalizedValue()
{
    // Default normalization (and the normalization buttons will use) is 0 to 1.0
    return (double)(physicalValue() - physicalMin()) / (double)(physicalMax() - physicalMin());
}

#pragma mark HIDGamepadButton

HIDInputType HIDGamepadButton::gamepadValueChanged(IOHIDValueRef value)
{
    valueChanged(value);
    m_value.setValue(normalizedValue());
    return m_value.value() > 0.5 ? HIDInputType::ButtonPress : HIDInputType::NotAButtonPress;
}

#pragma mark HIDGamepadAxis

HIDInputType HIDGamepadAxis::gamepadValueChanged(IOHIDValueRef value)
{
    valueChanged(value);
    m_value.setValue(normalizedValue());
    return HIDInputType::NotAButtonPress;
}

double HIDGamepadAxis::normalizedValue()
{
    // Web Gamepad axes have a value of -1.0 to 1.0
    return (HIDGamepadElement::normalizedValue() * 2.0) - 1.0;
}

#pragma mark HIDGamepadHatswitch

HIDInputType HIDGamepadHatswitch::gamepadValueChanged(IOHIDValueRef value)
{
    valueChanged(value);

    for (size_t i = 0; i < 4; ++i)
        m_buttonValues[i].setValue(0.0);

    switch (physicalValue()) {
    case 0:
        m_buttonValues[0].setValue(1.0);
        break;
    case 45:
        m_buttonValues[0].setValue(1.0);
        m_buttonValues[1].setValue(1.0);
        break;
    case 90:
        m_buttonValues[1].setValue(1.0);
        break;
    case 135:
        m_buttonValues[1].setValue(1.0);
        m_buttonValues[2].setValue(1.0);
        break;
    case 180:
        m_buttonValues[2].setValue(1.0);
        break;
    case 225:
        m_buttonValues[2].setValue(1.0);
        m_buttonValues[3].setValue(1.0);
        break;
    case 270:
        m_buttonValues[3].setValue(1.0);
        break;
    case 315:
        m_buttonValues[3].setValue(1.0);
        m_buttonValues[0].setValue(1.0);
        break;
    default:
        break;
    };

    return HIDInputType::ButtonPress;
}

double HIDGamepadHatswitch::normalizedValue()
{
    // Hatswitch normalizedValue should never be accessed.
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && PLATFORM(MAC)
