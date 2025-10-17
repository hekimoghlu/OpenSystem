/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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

#include "HIDElement.h"
#include "SharedGamepadValue.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

enum class HIDInputType {
    ButtonPress,
    NotAButtonPress,
};

class HIDGamepadElement : public HIDElement {
    WTF_MAKE_TZONE_ALLOCATED(HIDGamepadElement);
public:
    virtual ~HIDGamepadElement() { }

    virtual HIDInputType gamepadValueChanged(IOHIDValueRef) = 0;

    void refreshCurrentValue();

protected:
    HIDGamepadElement(const HIDElement&);

    virtual double normalizedValue();
    virtual bool isButton() const { return false; }
    virtual bool isAxis() const { return false; }
};

class HIDGamepadButton final : public HIDGamepadElement {
public:
    HIDGamepadButton(const HIDElement& element, SharedGamepadValue& value)
        : HIDGamepadElement(element)
        , m_value(value)
    {
    }

    bool isButton() const final { return true; }

private:
    HIDInputType gamepadValueChanged(IOHIDValueRef) override;

    SharedGamepadValue m_value;
};

class HIDGamepadAxis final : public HIDGamepadElement {
public:
    HIDGamepadAxis(const HIDElement& element, SharedGamepadValue& value)
        : HIDGamepadElement(element)
        , m_value(value)
    {
    }

    bool isAxis() const final { return true; }

private:
    HIDInputType gamepadValueChanged(IOHIDValueRef) override;
    double normalizedValue() final;

    SharedGamepadValue m_value;
};

class HIDGamepadHatswitch : public HIDGamepadElement {
public:
    HIDGamepadHatswitch(const HIDElement& element, Vector<SharedGamepadValue>&& buttonValues)
        : HIDGamepadElement(element)
        , m_buttonValues(WTFMove(buttonValues))
    {
    }

    // Treat hatswitch value changes as a button press
    bool isButton() const final { return true; }

private:
    HIDInputType gamepadValueChanged(IOHIDValueRef) final;
    double normalizedValue() final;

    Vector<SharedGamepadValue> m_buttonValues;
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && PLATFORM(MAC)
