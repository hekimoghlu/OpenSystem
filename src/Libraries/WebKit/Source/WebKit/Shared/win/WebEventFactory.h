/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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

#include "WebKeyboardEvent.h"
#include "WebMouseEvent.h"
#include "WebWheelEvent.h"

#if ENABLE(TOUCH_EVENTS)
#include "WebTouchEvent.h"
#endif

#include <windows.h>

namespace WebKit {

class WebEventFactory {
public:
    static WebMouseEvent createWebMouseEvent(HWND, UINT message, WPARAM, LPARAM, bool didActivateWebView, float deviceScaleFactor);
    static WebWheelEvent createWebWheelEvent(HWND, UINT message, WPARAM, LPARAM, float deviceScaleFactor);
    static WebKeyboardEvent createWebKeyboardEvent(HWND, UINT message, WPARAM, LPARAM);
#if ENABLE(TOUCH_EVENTS)
    static WebTouchEvent createWebTouchEvent();
#endif
};

inline MSG createNativeEvent(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) { return { hwnd, message, wParam, lParam, 0, { } }; }

} // namespace WebKit
