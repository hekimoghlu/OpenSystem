/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 17, 2021.
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
#include "WKPagePrivateWPE.h"

#include "NativeWebKeyboardEvent.h"
#include "NativeWebMouseEvent.h"
#include "WKAPICast.h"
#include "WebPageProxy.h"
#include <wpe/wpe.h>

void WKPageHandleKeyboardEvent(WKPageRef pageRef, WKKeyboardEvent event)
{
    using WebKit::NativeWebKeyboardEvent;

    wpe_input_keyboard_event wpeEvent;
    wpeEvent.time = 0;
    wpeEvent.key_code = event.keyCode;
    wpeEvent.hardware_key_code = event.hardwareKeyCode;
    wpeEvent.modifiers = event.modifiers;
    switch (event.type) {
    case kWKEventKeyDown:
        wpeEvent.pressed = true;
        break;
    case kWKEventKeyUp:
        wpeEvent.pressed = false;
        break;
    default:
        ASSERT_NOT_REACHED();
        return;
    }

    NativeWebKeyboardEvent::HandledByInputMethod handledByInputMethod = NativeWebKeyboardEvent::HandledByInputMethod::No;
    std::optional<Vector<WebCore::CompositionUnderline>> preeditUnderlines;
    std::optional<WebKit::EditingRange> preeditSelectionRange;
    WebKit::toImpl(pageRef)->handleKeyboardEvent(NativeWebKeyboardEvent(&wpeEvent, std::span { event.text, event.length }, false, handledByInputMethod, WTFMove(preeditUnderlines), WTFMove(preeditSelectionRange)));
}

void WKPageHandleMouseEvent(WKPageRef pageRef, WKMouseEvent event)
{
    using WebKit::NativeWebMouseEvent;

    wpe_input_pointer_event wpeEvent;

    switch (event.type) {
    case kWKEventMouseDown:
        wpeEvent.type = wpe_input_pointer_event_type_button;
        wpeEvent.state = 1;
        break;
    case kWKEventMouseUp:
        wpeEvent.type = wpe_input_pointer_event_type_button;
        wpeEvent.state = 0;
        break;
    case kWKEventMouseMove:
        wpeEvent.type = wpe_input_pointer_event_type_motion;
        wpeEvent.state = 0;
        break;
    default:
        ASSERT_NOT_REACHED();
        return;
    }

    switch (event.button) {
    case kWKEventMouseButtonLeftButton:
        wpeEvent.button = 1;
        break;
    case kWKEventMouseButtonMiddleButton:
        wpeEvent.button = 3;
        break;
    case kWKEventMouseButtonRightButton:
        wpeEvent.button = 2;
        break;
    case kWKEventMouseButtonNoButton:
        wpeEvent.button = 0;
        break;
    default:
        ASSERT_NOT_REACHED();
        return;
    }
    wpeEvent.time = 0;
    wpeEvent.x = event.position.x;
    wpeEvent.y = event.position.y;
    wpeEvent.modifiers = event.modifiers;

    const float deviceScaleFactor = 1;

    WebKit::toImpl(pageRef)->handleMouseEvent(NativeWebMouseEvent(&wpeEvent, deviceScaleFactor));
}
