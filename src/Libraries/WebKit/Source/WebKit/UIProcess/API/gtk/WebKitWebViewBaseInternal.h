/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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

#include <WebKit/WKBase.h>
#include <WebCore/PlatformMouseEvent.h>
#include <wtf/text/WTFString.h>

#if USE(CAIRO)
typedef struct _cairo_surface cairo_surface_t;
#elif USE(SKIA)
class SkImage;
#endif

typedef struct _WebKitWebViewBase WebKitWebViewBase;

struct KeyEvent {
    unsigned type { 0 };
    unsigned keyval { 0 };
    unsigned modifiers { 0 };
};

enum class MouseEventType { Press, Release, Motion };
WK_EXPORT void webkitWebViewBaseSynthesizeMouseEvent(WebKitWebViewBase*, MouseEventType type, unsigned button, unsigned short buttons, int x, int y, unsigned modifiers, int clickCount, const String& pointerType = "mouse"_s, WebCore::PlatformMouseEvent::IsTouch isTouchEvent = WebCore::PlatformMouseEvent::IsTouch::No);

enum class KeyEventType { Press, Release, Insert };
enum class ShouldTranslateKeyboardState : bool { No, Yes };
WK_EXPORT void webkitWebViewBaseSynthesizeKeyEvent(WebKitWebViewBase*, KeyEventType, unsigned keyVal, unsigned modifiers, ShouldTranslateKeyboardState);

enum class WheelEventPhase { NoPhase, Began, Changed, Ended, Cancelled, MayBegin };
WK_EXPORT void webkitWebViewBaseSynthesizeWheelEvent(WebKitWebViewBase*, double deltaX, double deltaY, int x, int y, WheelEventPhase, WheelEventPhase momentumPhase, bool);

#if USE(CAIRO)
WK_EXPORT cairo_surface_t* webkitWebViewBaseSnapshotForTesting(WebKitWebViewBase*);
#elif USE(SKIA)
WK_EXPORT SkImage* webkitWebViewBaseSnapshotForTesting(WebKitWebViewBase*);
#endif
