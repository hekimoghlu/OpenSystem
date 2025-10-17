/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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
#include "WebXRGamepad.h"

#if ENABLE(WEBXR) && ENABLE(GAMEPAD)

#include "Gamepad.h"
#include "GamepadConstants.h"
#include "XRTargetRayMode.h"

namespace WebCore {

// https://immersive-web.github.io/webxr-gamepads-module/#gamepad-differences
// Gamepad's index attribute must be -1.
constexpr int DefaultXRGamepadId = -1;

// https://immersive-web.github.io/webxr-gamepads-module/#gamepad-differences
WebXRGamepad::WebXRGamepad(double timestamp, double connectTime, const PlatformXR::FrameData::InputSource& source)
    : PlatformGamepad(DefaultXRGamepadId)
{
    m_lastUpdateTime = MonotonicTime::fromRawSeconds(Seconds::fromMilliseconds(timestamp).value());
    m_connectTime = MonotonicTime::fromRawSeconds(Seconds::fromMilliseconds(connectTime).value());
    // In order to report a mapping of "xr-standard" the device MUST report a targetRayMode of "tracked-pointer" and MUST have a non-null gripSpace.
    // It MUST have at least one primary trigger, separate from any touchpads or thumbsticks
    if (source.targetRayMode == XRTargetRayMode::TrackedPointer && !source.buttons.isEmpty() && source.gripOrigin)
        m_mapping = xrStandardGamepadMappingString();
    m_axes = source.axes.map([](auto value) {
        return SharedGamepadValue(value);
    });
    m_buttons = source.buttons.map([](auto& value) {
        return SharedGamepadValue(value.pressedValue);
    });
}

} // namespace WebCore

#endif // ENABLE(WEBXR) && ENABLE(GAMEPAD)
