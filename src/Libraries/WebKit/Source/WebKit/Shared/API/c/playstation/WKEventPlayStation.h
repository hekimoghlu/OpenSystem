/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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
#include <WebKit/WKEvent.h>
#include <WebKit/WKGeometry.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    kWKEventNoType = -1,

    // WebMouseEvent
    kWKEventMouseDown,
    kWKEventMouseUp,
    kWKEventMouseMove,
    kWKEventMouseForceChanged,
    kWKEventMouseForceDown,
    kWKEventMouseForceUp,

    // WebWheelEvent
    kWKEventWheel,

    // WebKeyboardEvent
    kWKEventKeyDown,
    kWKEventKeyUp
};
typedef int32_t WKEventType;

struct WKKeyboardEvent {
    WKEventType type;
    const char* text;
    int32_t length;
    const char* keyIdentifier;
    uint32_t modifiers;
    int32_t virtualKeyCode;
    int32_t caretOffset;
};
typedef struct WKKeyboardEvent WKKeyboardEvent;

enum {
    kWKInputTypeNormal,
    kWKInputTypeSetComposition,
    kWKInputTypeConfirmComposition,
    kWKInputTypeCancelComposition,
};
typedef uint8_t WKInputType;

struct WKMouseEvent {
    WKEventType type;
    WKEventMouseButton button;
    WKPoint position;
    int32_t clickCount;
    uint32_t modifiers;
};
typedef struct WKMouseEvent WKMouseEvent;

struct WKWheelEvent {
    WKEventType type;
    WKPoint position;
    WKSize delta;
    WKSize wheelTicks;
    uint32_t modifiers;
};
typedef struct WKWheelEvent WKWheelEvent;

WK_EXPORT WKKeyboardEvent WKKeyboardEventMake(WKEventType type, WKInputType inputType, const char* text, uint32_t length, const char* keyIdentifier, int32_t virtualKeyCode, int32_t caretOffset, uint32_t attributes, uint32_t modifiers);

WK_EXPORT WKMouseEvent WKMouseEventMake(WKEventType type, WKEventMouseButton button, WKPoint position, int32_t clickCount, uint32_t modifiers);

WK_EXPORT WKWheelEvent WKWheelEventMake(WKEventType type, WKPoint position, WKSize delta, WKSize wheelTicks, uint32_t modifiers);

#ifdef __cplusplus
}
#endif
