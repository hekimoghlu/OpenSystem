/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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
#import "config.h"
#import "MouseEvent.h"
#import "PlatformMouseEvent.h"

#if ENABLE(TOUCH_EVENTS) && PLATFORM(IOS_FAMILY)

#import "EventNames.h"

namespace WebCore {

static AtomString mouseEventType(PlatformTouchPoint::TouchPhaseType phase)
{
    switch (phase) {
    case PlatformTouchPoint::TouchPhaseBegan:
        return eventNames().mousedownEvent;
    case PlatformTouchPoint::TouchPhaseMoved:
    case PlatformTouchPoint::TouchPhaseStationary:
        return eventNames().mousemoveEvent;
    case PlatformTouchPoint::TouchPhaseEnded:
    case PlatformTouchPoint::TouchPhaseCancelled:
        return eventNames().mouseupEvent;
    }
    ASSERT_NOT_REACHED();
    return nullAtom();
}

Ref<MouseEvent> MouseEvent::create(const PlatformTouchEvent& event, unsigned index, Ref<WindowProxy>&& view, IsCancelable cancelable)
{
    return adoptRef(*new MouseEvent(EventInterfaceType::MouseEvent, mouseEventType(event.touchPhaseAtIndex(index)), CanBubble::Yes, cancelable, IsComposed::Yes,
        event.timestamp().approximateMonotonicTime(), WTFMove(view), 0, event.touchLocationInRootViewAtIndex(index), event.touchLocationInRootViewAtIndex(index), 0, 0,
        event.modifiers(), MouseButton::Left, 0, nullptr, 0, SyntheticClickType::NoTap, { }, { }, IsSimulated::No, IsTrusted::Yes));
}

} // namespace WebCore

#endif // ENABLE(TOUCH_EVENTS) && PLATFORM(IOS_FAMILY)
