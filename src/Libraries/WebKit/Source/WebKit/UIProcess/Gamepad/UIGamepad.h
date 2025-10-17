/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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

#if ENABLE(GAMEPAD)

#include <WebCore/GamepadHapticEffectType.h>
#include <WebCore/SharedGamepadValue.h>
#include <wtf/MonotonicTime.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class PlatformGamepad;
}

namespace WebKit {

class GamepadData;

class UIGamepad {
    WTF_MAKE_TZONE_ALLOCATED(UIGamepad);
public:
    UIGamepad(WebCore::PlatformGamepad&);

    unsigned index() const { return m_index; }

    GamepadData gamepadData() const;

    void updateFromPlatformGamepad(WebCore::PlatformGamepad&);

private:
    unsigned m_index;
    String m_id;
    String m_mapping;
    Vector<WebCore::SharedGamepadValue> m_axisValues;
    Vector<WebCore::SharedGamepadValue> m_buttonValues;
    MonotonicTime m_lastUpdateTime;
    WebCore::GamepadHapticEffectTypeSet m_supportedEffectTypes;
};

}

#endif // ENABLE(GAMEPAD)
