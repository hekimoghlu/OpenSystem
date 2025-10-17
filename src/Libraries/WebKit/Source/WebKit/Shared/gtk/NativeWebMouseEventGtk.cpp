/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#include "NativeWebMouseEvent.h"

#include "WebEventFactory.h"
#include <WebCore/GtkVersioning.h>
#include <gdk/gdk.h>

namespace WebKit {

#if USE(GTK4)
#define constructNativeEvent(event) event
#else
#define constructNativeEvent(event) gdk_event_copy(event)
#endif

NativeWebMouseEvent::NativeWebMouseEvent(GdkEvent* event, int eventClickCount, std::optional<WebCore::FloatSize> delta)
    : WebMouseEvent(WebEventFactory::createWebMouseEvent(event, eventClickCount, delta))
    , m_nativeEvent(constructNativeEvent(event))
{
}

NativeWebMouseEvent::NativeWebMouseEvent(GdkEvent* event, const WebCore::IntPoint& position, int eventClickCount, std::optional<WebCore::FloatSize> delta)
    : WebMouseEvent(WebEventFactory::createWebMouseEvent(event, position, position, eventClickCount, delta))
    , m_nativeEvent(constructNativeEvent(event))
{
}

NativeWebMouseEvent::NativeWebMouseEvent(const WebCore::IntPoint& position)
    : WebMouseEvent(WebEventFactory::createWebMouseEvent(position))
{
}

NativeWebMouseEvent::NativeWebMouseEvent(WebEventType type, WebMouseEventButton button, unsigned short buttons, const WebCore::IntPoint& position, const WebCore::IntPoint& globalPosition, int clickCount, OptionSet<WebEventModifier> modifiers, std::optional<WebCore::FloatSize> delta, WebCore::PointerID pointerId, const String& pointerType, WebCore::PlatformMouseEvent::IsTouch isTouchEvent)
    : WebMouseEvent(WebEvent(type, modifiers, WallTime::now()), button, buttons, position, globalPosition, delta.value_or(WebCore::FloatSize()).width(), delta.value_or(WebCore::FloatSize()).height(), 0, clickCount, 0, WebMouseEventSyntheticClickType::NoTap, isTouchEvent, pointerId, pointerType)
{
}

NativeWebMouseEvent::NativeWebMouseEvent(const NativeWebMouseEvent& event)
    : WebMouseEvent(WebEvent(event.type(), event.modifiers(), event.timestamp()), event.button(), event.buttons(), event.position(), event.globalPosition(), event.deltaX(), event.deltaY(), event.deltaZ(), event.clickCount(), 0, WebMouseEventSyntheticClickType::NoTap, event.isTouchEvent(), event.pointerId(), event.pointerType())
    , m_nativeEvent(event.nativeEvent() ? constructNativeEvent(const_cast<GdkEvent*>(event.nativeEvent())) : nullptr)
{
}

} // namespace WebKit

#undef constructNativeEvent
