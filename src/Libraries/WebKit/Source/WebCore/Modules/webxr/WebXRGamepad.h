/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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

#if ENABLE(WEBXR) && ENABLE(GAMEPAD)

#include "PlatformGamepad.h"
#include "PlatformXR.h"
#include <wtf/Vector.h>

namespace WebCore {

class WebXRGamepad: public PlatformGamepad {
public:
    WebXRGamepad(double timestamp, double connectTime, const PlatformXR::FrameData::InputSource&);

private:
    const Vector<SharedGamepadValue>& axisValues() const final { return m_axes; }
    const Vector<SharedGamepadValue>& buttonValues() const final { return m_buttons; }

    Vector<SharedGamepadValue> m_axes;
    Vector<SharedGamepadValue> m_buttons;
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
