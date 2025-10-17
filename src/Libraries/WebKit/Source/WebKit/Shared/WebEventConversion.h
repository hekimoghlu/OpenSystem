/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
#ifndef WebEventConversion_h
#define WebEventConversion_h

#include "WebEventModifier.h"
#include <WebCore/PlatformKeyboardEvent.h>
#include <WebCore/PlatformMouseEvent.h>
#include <WebCore/PlatformWheelEvent.h>

#if ENABLE(IOS_TOUCH_EVENTS)
#include <WebKitAdditions/PlatformTouchEventIOS.h>
#elif ENABLE(TOUCH_EVENTS)
#include <WebCore/PlatformTouchEvent.h>
#include <WebCore/PlatformTouchPoint.h>
#endif

#if ENABLE(MAC_GESTURE_EVENTS)
#include <WebKitAdditions/PlatformGestureEventMac.h>
#endif

namespace WebKit {

class WebMouseEvent;
class WebWheelEvent;
class WebKeyboardEvent;

#if ENABLE(TOUCH_EVENTS)
class WebTouchEvent;
class WebTouchPoint;
#endif

#if ENABLE(MAC_GESTURE_EVENTS)
class WebGestureEvent;
#endif

OptionSet<WebCore::PlatformEvent::Modifier> platform(OptionSet<WebEventModifier>);

WebCore::PlatformMouseEvent platform(const WebMouseEvent&);
WebCore::PlatformWheelEvent platform(const WebWheelEvent&);
WebCore::PlatformKeyboardEvent platform(const WebKeyboardEvent&);

#if ENABLE(TOUCH_EVENTS)
WebCore::PlatformTouchEvent platform(const WebTouchEvent&);
#if !ENABLE(IOS_TOUCH_EVENTS)
WebCore::PlatformTouchPoint platform(const WebTouchPoint&);
#endif
#endif

#if ENABLE(MAC_GESTURE_EVENTS)
WebCore::PlatformGestureEvent platform(const WebGestureEvent&);
#endif

} // namespace WebKit

#endif // WebEventConversion_h
