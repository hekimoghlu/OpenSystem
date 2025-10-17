/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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
#include "HIDGamepad.h"

#if ENABLE(GAMEPAD) && PLATFORM(MAC)

#include "Dualshock3HIDGamepad.h"
#include "GenericHIDGamepad.h"
#include "KnownGamepads.h"
#include "Logging.h"
#include "LogitechGamepad.h"
#include "StadiaHIDGamepad.h"
#include <IOKit/hid/IOHIDElement.h>
#include <IOKit/hid/IOHIDUsageTables.h>
#include <IOKit/hid/IOHIDValue.h>
#include <wtf/HexNumber.h>
#include <wtf/cf/TypeCastsCF.h>
#include <wtf/text/CString.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

std::unique_ptr<HIDGamepad> HIDGamepad::create(IOHIDDeviceRef rawDevice, unsigned index)
{
    auto device = HIDDevice { rawDevice };

    std::unique_ptr<HIDGamepad> newGamepad;

    switch ((KnownGamepad)device.fullProductIdentifier()) {
    case Dualshock3:
        newGamepad = makeUnique<Dualshock3HIDGamepad>(WTFMove(device), index);
        break;
    case LogitechF310:
    case LogitechF710:
        newGamepad = makeUnique<LogitechGamepad>(WTFMove(device), index);
        break;
    case StadiaA:
        newGamepad = makeUnique<StadiaHIDGamepad>(WTFMove(device), index);
        break;
    default:
        newGamepad = makeUnique<GenericHIDGamepad>(WTFMove(device), index);
    }

    newGamepad->initialize();
    return newGamepad;
}

HIDGamepad::HIDGamepad(HIDDevice&& device, unsigned index)
    : PlatformGamepad(index)
    , m_device(WTFMove(device))
{
    m_connectTime = m_lastUpdateTime = MonotonicTime::now();

    // Currently the spec has no formatting for the id string.
    // This string formatting matches Firefox.
    m_id = makeString(hex(m_device.vendorID(), Lowercase), '-', hex(m_device.productID(), Lowercase), '-', m_device.productName());
}

void HIDGamepad::initialize()
{
    for (auto& element : m_elementMap.values())
        element->refreshCurrentValue();
}

HIDInputType HIDGamepad::valueChanged(IOHIDValueRef value)
{
    IOHIDElementCookie cookie = IOHIDElementGetCookie(IOHIDValueGetElement(value));
    HIDGamepadElement* element = m_elementMap.get(cookie);

    // This might be an element we don't currently handle as input so we can skip it.
    if (!element)
        return HIDInputType::NotAButtonPress;

    m_lastUpdateTime = MonotonicTime::now();

    return element->gamepadValueChanged(value);
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && PLATFORM(MAC)
