/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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
#include "InputMethodFilter.h"

#include "WebKitInputMethodContextPrivate.h"
#include <WebCore/IntRect.h>
#include <wpe/wpe.h>

#if ENABLE(WPE_PLATFORM)
#include <wpe/wpe-platform.h>
#endif

namespace WebKit {
using namespace WebCore;

IntRect InputMethodFilter::platformTransformCursorRectToViewCoordinates(const IntRect& cursorRect)
{
    return cursorRect;
}

bool InputMethodFilter::platformEventKeyIsKeyPress(PlatformEventKey* event) const
{
#if ENABLE(WPE_PLATFORM)
    if (m_useWPEPlatformEvents) {
        auto* wpe_platform_event = static_cast<WPEEvent*>(event);
        return wpe_event_get_event_type(wpe_platform_event) == WPE_EVENT_KEYBOARD_KEY_DOWN;
    }
#endif
    auto* wpe_event = static_cast<struct wpe_input_keyboard_event*>(event);
    return wpe_event->pressed;
}

} // namespace WebKit
