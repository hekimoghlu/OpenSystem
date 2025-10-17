/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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

#include "ArgumentCoders.h"

#if HAVE(PER_APP_ACCESSIBILITY_PREFERENCES)
#include "AccessibilitySupportSPI.h"
#endif

namespace WebKit {

#if HAVE(PER_APP_ACCESSIBILITY_PREFERENCES)
enum class WebKitAXValueState : int {
    AXValueStateInvalid = -2,
    AXValueStateEmpty = -1,
    AXValueStateOff,
    AXValueStateOn
};

inline WebKitAXValueState toWebKitAXValueState(AXValueState value)
{
    switch (value) {
    case AXValueState::AXValueStateInvalid:
        return WebKitAXValueState::AXValueStateInvalid;
    case AXValueState::AXValueStateEmpty:
        return WebKitAXValueState::AXValueStateEmpty;
    case AXValueState::AXValueStateOff:
        return WebKitAXValueState::AXValueStateOff;
    case AXValueState::AXValueStateOn:
        return WebKitAXValueState::AXValueStateOn;
    }

    ASSERT_NOT_REACHED();
    return WebKitAXValueState::AXValueStateInvalid;
}

inline AXValueState fromWebKitAXValueState(WebKitAXValueState value)
{
    switch (value) {
    case WebKitAXValueState::AXValueStateInvalid:
        return AXValueState::AXValueStateInvalid;
    case WebKitAXValueState::AXValueStateEmpty:
        return AXValueState::AXValueStateEmpty;
    case WebKitAXValueState::AXValueStateOff:
        return AXValueState::AXValueStateOff;
    case WebKitAXValueState::AXValueStateOn:
        return AXValueState::AXValueStateOn;
    }

    ASSERT_NOT_REACHED();
    return AXValueState::AXValueStateInvalid;
}
#endif

struct AccessibilityPreferences {
#if HAVE(PER_APP_ACCESSIBILITY_PREFERENCES)
    WebKitAXValueState reduceMotionEnabled { WebKitAXValueState::AXValueStateEmpty };
    WebKitAXValueState increaseButtonLegibility { WebKitAXValueState::AXValueStateEmpty };
    WebKitAXValueState enhanceTextLegibility { WebKitAXValueState::AXValueStateEmpty };
    WebKitAXValueState darkenSystemColors { WebKitAXValueState::AXValueStateEmpty };
    WebKitAXValueState invertColorsEnabled { WebKitAXValueState::AXValueStateEmpty };

#endif
    bool imageAnimationEnabled { true };
    bool enhanceTextLegibilityOverall { false };
#if ENABLE(ACCESSIBILITY_NON_BLINKING_CURSOR)
    bool prefersNonBlinkingCursor { false };
#endif
};

} // namespace WebKit
