/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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
#include <WebKit/WKGeometry.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    kWKCursorTypePointer = 0,
    kWKCursorTypeHand = 2,
    kWKCursorTypeNone = 37,
};
typedef uint32_t WKCursorType;

typedef void(*WKViewSetViewNeedsDisplayCallback)(WKViewRef view, WKRect rect, const void* clientInfo);
typedef void(*WKViewEnterFullScreen)(WKViewRef view, const void* clientInfo);
typedef void(*WKViewExitFullScreen)(WKViewRef view, const void* clientInfo);
typedef void(*WKViewCloseFullScreen)(WKViewRef view, const void* clientInfo);
typedef void(*WKViewBeganEnterFullScreen)(WKViewRef view, const WKRect initialFrame, const WKRect finalFrame, const void* clientInfo);
typedef void(*WKViewBeganExitFullScreen)(WKViewRef view, const WKRect initialFrame, const WKRect finalFrame, const void* clientInfo);
typedef void(*WKViewSetCursorCallback)(WKViewRef view, WKCursorType cursorType, const void* clientInfo);

typedef struct WKViewClientBase {
    int version;
    const void* clientInfo;
} WKViewClientBase;

typedef struct WKViewClientV0 {
    WKViewClientBase base;

    // version 0
    WKViewSetViewNeedsDisplayCallback setViewNeedsDisplay;
    WKViewEnterFullScreen enterFullScreen;
    WKViewExitFullScreen exitFullScreen;
    WKViewCloseFullScreen closeFullScreen;
    WKViewBeganEnterFullScreen beganEnterFullScreen;
    WKViewBeganExitFullScreen beganExitFullScreen;
    WKViewSetCursorCallback setCursor;
} WKViewClientV0;

#ifdef __cplusplus
}
#endif
