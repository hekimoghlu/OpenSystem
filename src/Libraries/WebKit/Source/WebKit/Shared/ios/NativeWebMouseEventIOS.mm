/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 18, 2024.
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
#import "NativeWebMouseEvent.h"

#if PLATFORM(IOS_FAMILY)

#import "WebIOSEventFactory.h"

namespace WebKit {

NativeWebMouseEvent::NativeWebMouseEvent(::WebEvent *event)
    : WebMouseEvent(WebIOSEventFactory::createWebMouseEvent(event))
    , m_nativeEvent(event)
{
}

NativeWebMouseEvent::NativeWebMouseEvent(WebEventType type, WebMouseEventButton button, unsigned short buttons, const WebCore::IntPoint& position, const WebCore::IntPoint& globalPosition, float deltaX, float deltaY, float deltaZ, int clickCount, OptionSet<WebEventModifier> modifiers, WallTime timestamp, double force, GestureWasCancelled gestureWasCancelled, const String& pointerType)
    : WebMouseEvent({ type, modifiers, timestamp }, button, buttons, position, globalPosition, deltaX, deltaY, deltaZ, clickCount, force, WebMouseEventSyntheticClickType::NoTap, WebCore::mousePointerID, pointerType, gestureWasCancelled)
{
}

NativeWebMouseEvent::NativeWebMouseEvent(const NativeWebMouseEvent& otherEvent, const WebCore::IntPoint& position, const WebCore::IntPoint& globalPosition, float deltaX, float deltaY, float deltaZ)
    : WebMouseEvent({ otherEvent.type(), otherEvent.modifiers(), otherEvent.timestamp() }, otherEvent.button(), otherEvent.buttons(), position, globalPosition, deltaX, deltaY, deltaZ, otherEvent.clickCount(), otherEvent.force(), otherEvent.syntheticClickType(), otherEvent.pointerId(), otherEvent.pointerType(), otherEvent.gestureWasCancelled())
{
}

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
