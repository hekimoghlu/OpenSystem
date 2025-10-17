/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#include "WheelEvent.h"

#include "DataTransfer.h"
#include "EventNames.h"
#include "Node.h"
#include "PlatformMouseEvent.h"
#include <wtf/MathExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WheelEvent);

inline static unsigned determineDeltaMode(const PlatformWheelEvent& event)
{
    return event.granularity() == ScrollByPageWheelEvent ? WheelEvent::DOM_DELTA_PAGE : WheelEvent::DOM_DELTA_PIXEL;
}

static double wheelDeltaToDelta(int wheelDelta)
{
    // Avoid returning negative zero.
    if (!wheelDelta)
        return 0;
    return -static_cast<double>(wheelDelta);
}

inline WheelEvent::WheelEvent()
    : MouseEvent(EventInterfaceType::WheelEvent)
{
}

inline WheelEvent::WheelEvent(const AtomString& type, const Init& initializer)
    : MouseEvent(EventInterfaceType::WheelEvent, type, initializer, IsTrusted::No)
    , m_wheelDelta(initializer.wheelDeltaX ? initializer.wheelDeltaX : clampTo<int>(-initializer.deltaX), initializer.wheelDeltaY ? initializer.wheelDeltaY : clampTo<int>(-initializer.deltaY))
    , m_deltaX(initializer.deltaX ? initializer.deltaX : wheelDeltaToDelta(initializer.wheelDeltaX))
    , m_deltaY(initializer.deltaY ? initializer.deltaY : wheelDeltaToDelta(initializer.wheelDeltaY))
    , m_deltaZ(initializer.deltaZ)
    , m_deltaMode(initializer.deltaMode)
{
}

inline WheelEvent::WheelEvent(const PlatformWheelEvent& event, RefPtr<WindowProxy>&& view, IsCancelable isCancelable)
    : MouseEvent(EventInterfaceType::WheelEvent, eventNames().wheelEvent, CanBubble::Yes, isCancelable, IsComposed::Yes, event.timestamp().approximateMonotonicTime(), WTFMove(view), 0,
        event.globalPosition(), event.position() , 0, 0, event.modifiers(), MouseButton::Left, 0, nullptr, 0, SyntheticClickType::NoTap, { }, { }, IsSimulated::No, IsTrusted::Yes)
    , m_wheelDelta(event.wheelTicksX() * TickMultiplier, event.wheelTicksY() * TickMultiplier)
    , m_deltaX(-event.deltaX())
    , m_deltaY(-event.deltaY())
    , m_deltaMode(determineDeltaMode(event))
    , m_underlyingPlatformEvent(event)
{
}

Ref<WheelEvent> WheelEvent::create(const PlatformWheelEvent& event, RefPtr<WindowProxy>&& view, IsCancelable isCancelable)
{
    return adoptRef(*new WheelEvent(event, WTFMove(view), isCancelable));
}

Ref<WheelEvent> WheelEvent::createForBindings()
{
    return adoptRef(*new WheelEvent);
}

Ref<WheelEvent> WheelEvent::create(const AtomString& type, const Init& initializer)
{
    return adoptRef(*new WheelEvent(type, initializer));
}

bool WheelEvent::isWheelEvent() const
{
    return true;
}

} // namespace WebCore
