/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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
#import "NativeWebGestureEvent.h"

#if ENABLE(MAC_GESTURE_EVENTS)

#import "WebGestureEvent.h"
#import <WebCore/IntPoint.h>
#import <WebCore/PlatformEventFactoryMac.h>

namespace WebKit {

static inline std::optional<WebEventType> webEventTypeForNSEvent(NSEvent *event)
{
    switch (event.phase) {
    case NSEventPhaseBegan:
        return WebEventType::GestureStart;
    case NSEventPhaseChanged:
        return WebEventType::GestureChange;
    case NSEventPhaseEnded:
    case NSEventPhaseCancelled:
        return WebEventType::GestureEnd;
    default:
        break;
    }
    return std::nullopt;
}

static NSPoint pointForEvent(NSEvent *event, NSView *windowView)
{
    NSPoint location = [event locationInWindow];
    if (windowView)
        location = [windowView convertPoint:location fromView:nil];
    return location;
}

std::optional<NativeWebGestureEvent> NativeWebGestureEvent::create(NSEvent *event, NSView *view)
{
    auto type = webEventTypeForNSEvent(event);
    if (!type)
        return std::nullopt;
    return { NativeWebGestureEvent { *type, event, view } };
}

NativeWebGestureEvent::NativeWebGestureEvent(WebEventType type, NSEvent *event, NSView *view)
    : WebGestureEvent(
        { type, OptionSet<WebEventModifier> { }, WebCore::eventTimeStampSince1970(event.timestamp) },
        WebCore::IntPoint(pointForEvent(event, view)),
        event.type == NSEventTypeMagnify ? event.magnification : 0,
        event.type == NSEventTypeRotate ? event.rotation : 0)
    , m_nativeEvent(event)
{
}

} // namespace WebKit

#endif // ENABLE(MAC_GESTURE_EVENTS)
