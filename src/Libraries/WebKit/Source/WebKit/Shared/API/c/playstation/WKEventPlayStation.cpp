/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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
#include "WKEventPlayStation.h"

WKKeyboardEvent WKKeyboardEventMake(WKEventType type, WKInputType inputType, const char* text, uint32_t length, const char* keyIdentifier, int32_t virtualKeyCode, int32_t caretOffset, uint32_t attributes, uint32_t modifiers)
{
    WKKeyboardEvent keyboardEvent;
    keyboardEvent.type = type;
    keyboardEvent.virtualKeyCode = virtualKeyCode;
    keyboardEvent.modifiers = modifiers;
    keyboardEvent.caretOffset = caretOffset;

    // see http://www.w3.org/TR/DOM-Level-3-Events/#keys-IME
    if (inputType == kWKInputTypeSetComposition)
        keyboardEvent.keyIdentifier = "Convert";
    else if (inputType == kWKInputTypeConfirmComposition)
        keyboardEvent.keyIdentifier = "Accept";
    else if (inputType == kWKInputTypeCancelComposition)
        keyboardEvent.keyIdentifier = "Cancel";
    else
        keyboardEvent.keyIdentifier = keyIdentifier;

    if (length > 0) {
        keyboardEvent.text = text;
        keyboardEvent.length = length;
    } else {
        keyboardEvent.text = nullptr;
        keyboardEvent.length = 0;
    }

    return keyboardEvent;
}

WKMouseEvent WKMouseEventMake(WKEventType type, WKEventMouseButton button, WKPoint position, int32_t clickCount, uint32_t modifiers)
{
    WKMouseEvent mouseEvent;
    mouseEvent.type = type;
    mouseEvent.button = button;
    mouseEvent.position = position;
    mouseEvent.clickCount = clickCount;
    mouseEvent.modifiers = modifiers;
    return mouseEvent;
}

WKWheelEvent WKWheelEventMake(WKEventType type, WKPoint position, WKSize delta, WKSize wheelTicks, uint32_t modifiers)
{
    WKWheelEvent wheelEvent;
    wheelEvent.type = type;
    wheelEvent.position = position;
    wheelEvent.delta = delta;
    wheelEvent.wheelTicks = wheelTicks;
    wheelEvent.modifiers = modifiers;
    return wheelEvent;
}
