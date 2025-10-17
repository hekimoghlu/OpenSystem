/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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

#if ENABLE(GAMEPAD) && OS(LINUX)

#include "PlatformGamepad.h"

#include <libmanette.h>
#include <wtf/HashMap.h>
#include <wtf/glib/GRefPtr.h>

namespace WebCore {

class ManetteGamepad final : public PlatformGamepad {
public:
    // Refer https://www.w3.org/TR/gamepad/#gamepadbutton-interface
    enum class StandardGamepadAxis : int8_t {
        Unknown = -1,
        LeftStickX,
        LeftStickY,
        RightStickX,
        RightStickY,
        Count,
    };
    enum class StandardGamepadButton : int8_t {
        Unknown = -1,
        A,
        B,
        X,
        Y,
        LeftShoulder,
        RightShoulder,
        LeftTrigger,
        RightTrigger,
        Select,
        Start,
        LeftStick,
        RightStick,
        DPadUp,
        DPadDown,
        DPadLeft,
        DPadRight,
        Mode,
        Count,
    };

    ManetteGamepad(ManetteDevice*, unsigned index);
    virtual ~ManetteGamepad();

    const Vector<SharedGamepadValue>& axisValues() const final { return m_axisValues; }
    const Vector<SharedGamepadValue>& buttonValues() const final { return m_buttonValues; }

    void absoluteAxisChanged(ManetteDevice*, StandardGamepadAxis, double value);
    void buttonPressedOrReleased(ManetteDevice*, StandardGamepadButton, bool pressed);

private:
    GRefPtr<ManetteDevice> m_device;

    Vector<SharedGamepadValue> m_buttonValues;
    Vector<SharedGamepadValue> m_axisValues;
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && OS(LINUX)
