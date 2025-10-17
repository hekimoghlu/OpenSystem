/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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

#if ENABLE(TOUCH_EVENTS)

#include "TouchEvent.h"

#include "EventDispatcher.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(TouchEvent);

TouchEvent::TouchEvent()
    : MouseRelatedEvent(EventInterfaceType::TouchEvent)
{
}

TouchEvent::TouchEvent(TouchList* touches, TouchList* targetTouches, TouchList* changedTouches, const AtomString& type,
    RefPtr<WindowProxy>&& view, const IntPoint& globalLocation, OptionSet<Modifier> modifiers)
    : MouseRelatedEvent(EventInterfaceType::TouchEvent, type, IsCancelable::Yes, MonotonicTime::now(), WTFMove(view), globalLocation, modifiers)
    , m_touches(touches)
    , m_targetTouches(targetTouches)
    , m_changedTouches(changedTouches)
{
}

TouchEvent::TouchEvent(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
    : MouseRelatedEvent(EventInterfaceType::TouchEvent, type, initializer, isTrusted)
    , m_touches(initializer.touches ? initializer.touches : TouchList::create())
    , m_targetTouches(initializer.targetTouches ? initializer.targetTouches : TouchList::create())
    , m_changedTouches(initializer.changedTouches ? initializer.changedTouches : TouchList::create())
{
}

TouchEvent::~TouchEvent() = default;

bool TouchEvent::isTouchEvent() const
{
    return true;
}

} // namespace WebCore

#endif // ENABLE(TOUCH_EVENTS)
