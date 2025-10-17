/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#pragma once

#if ENABLE(GAMEPAD) && PLATFORM(MAC)

#include "HIDDevice.h"
#include "HIDGamepadElement.h"
#include "PlatformGamepad.h"
#include <IOKit/hid/IOHIDDevice.h>
#include <wtf/HashMap.h>

namespace WebCore {

class HIDGamepad : public PlatformGamepad {
public:
    static std::unique_ptr<HIDGamepad> create(IOHIDDeviceRef, unsigned index);

    const HIDDevice& hidDevice() const { return m_device; }

    void initialize();
    HIDInputType valueChanged(IOHIDValueRef);

    const Vector<SharedGamepadValue>& axisValues() const final { return m_axisValues; }
    const Vector<SharedGamepadValue>& buttonValues() const final { return m_buttonValues; }

    ASCIILiteral source() const final { return "HID"_s; }

protected:
    HIDGamepad(HIDDevice&&, unsigned index);

    HashMap<IOHIDElementCookie, std::unique_ptr<HIDGamepadElement>> m_elementMap;
    Vector<SharedGamepadValue> m_buttonValues;
    Vector<SharedGamepadValue> m_axisValues;

private:
    HIDDevice m_device;
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && PLATFORM(MAC)
