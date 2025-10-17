/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 10, 2025.
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
#include "NativeWebWheelEvent.h"

#include "WebEventFactory.h"
#include <WebCore/GtkVersioning.h>

#if USE(GTK4)
#define constructNativeEvent(event) event
#else
#define constructNativeEvent(event) gdk_event_copy(event)
#endif

namespace WebKit {

NativeWebWheelEvent::NativeWebWheelEvent(GdkEvent* event, const WebCore::IntPoint& position, const WebCore::IntPoint& globalPosition, const WebCore::FloatSize& delta, const WebCore::FloatSize& wheelTicks, WebWheelEvent::Phase phase, WebWheelEvent::Phase momentumPhase, bool hasPreciseDeltas)
    : WebWheelEvent(WebEventFactory::createWebWheelEvent(event, position, globalPosition, delta, wheelTicks, phase, momentumPhase, hasPreciseDeltas))
    , m_nativeEvent(event ? constructNativeEvent(event) : nullptr)
{
}

NativeWebWheelEvent::NativeWebWheelEvent(const NativeWebWheelEvent& event)
    : WebWheelEvent({ event.type(), event.modifiers(), event.timestamp() }, event.position(), event.globalPosition(), event.delta(), event.wheelTicks(), event.granularity(), event.phase(), event.momentumPhase(), event.hasPreciseScrollingDeltas())
    , m_nativeEvent(event.nativeEvent() ? constructNativeEvent(event.nativeEvent()) : nullptr)
{
}

} // namespace WebKit

#undef constructNativeEvent
