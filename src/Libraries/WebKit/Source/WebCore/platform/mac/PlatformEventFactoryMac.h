/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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

#include "PlatformKeyboardEvent.h"
#include "PlatformMouseEvent.h"
#include "PlatformWheelEvent.h"

#if PLATFORM(MAC)

namespace WebCore {

class PlatformEventFactory {
public:
    WEBCORE_EXPORT static PlatformMouseEvent createPlatformMouseEvent(NSEvent *, NSEvent *correspondingPressureEvent, NSView *windowView);
    static PlatformWheelEvent createPlatformWheelEvent(NSEvent *, NSView *windowView);
    WEBCORE_EXPORT static PlatformKeyboardEvent createPlatformKeyboardEvent(NSEvent *);
};

#if PLATFORM(COCOA) && defined(__OBJC__)

// FIXME: This function doesn't really belong in this header.
WEBCORE_EXPORT NSPoint globalPoint(const NSPoint& windowPoint, NSWindow *);

// FIXME: WebKit2 has a lot of code copied and pasted from PlatformEventFactoryMac in WebEventFactory. More of it should be shared with WebCore.
WEBCORE_EXPORT int windowsKeyCodeForKeyEvent(NSEvent *);
WEBCORE_EXPORT String keyIdentifierForKeyEvent(NSEvent *);
WEBCORE_EXPORT String keyForKeyEvent(NSEvent *);
WEBCORE_EXPORT String codeForKeyEvent(NSEvent *);
WEBCORE_EXPORT WallTime eventTimeStampSince1970(NSTimeInterval);
WEBCORE_EXPORT IntPoint unadjustedMovementForEvent(NSEvent *);

WEBCORE_EXPORT OptionSet<PlatformEvent::Modifier> modifiersForEvent(NSEvent *);
WEBCORE_EXPORT void getWheelEventDeltas(NSEvent *, float& deltaX, float& deltaY, BOOL& continuous);
WEBCORE_EXPORT UInt8 keyCharForEvent(NSEvent *);

#endif

} // namespace WebCore

#endif // PLATFORM(MAC)
