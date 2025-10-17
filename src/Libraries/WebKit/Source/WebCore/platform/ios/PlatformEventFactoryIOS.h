/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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

#if PLATFORM(IOS_FAMILY)

#include "PlatformKeyboardEvent.h"
#include "PlatformMouseEvent.h"
#include "PlatformWheelEvent.h"

#if USE(APPLE_INTERNAL_SDK)
#include <WebKitAdditions/PlatformTouchEventIOS.h>
#endif

OBJC_CLASS WebEvent;

namespace WebCore {

class PlatformEventFactory {
public:
    WEBCORE_EXPORT static PlatformMouseEvent createPlatformMouseEvent(WebEvent *);
    WEBCORE_EXPORT static PlatformWheelEvent createPlatformWheelEvent(WebEvent *);
    WEBCORE_EXPORT static PlatformKeyboardEvent createPlatformKeyboardEvent(WebEvent *);
#if ENABLE(TOUCH_EVENTS) || ENABLE(IOS_TOUCH_EVENTS)
    static PlatformTouchEvent createPlatformTouchEvent(WebEvent *);
    static PlatformTouchEvent createPlatformSimulatedTouchEvent(PlatformEvent::Type, IntPoint location);
#endif
};

WEBCORE_EXPORT String keyForKeyEvent(WebEvent *);
WEBCORE_EXPORT String codeForKeyEvent(WebEvent *);
WEBCORE_EXPORT String keyIdentifierForKeyEvent(WebEvent *);
WEBCORE_EXPORT int windowsKeyCodeForKeyEvent(WebEvent*);

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
