/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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
#include "ToggleEvent.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ToggleEvent);

ToggleEvent::ToggleEvent()
    : Event(EventInterfaceType::ToggleEvent)
{
}

ToggleEvent::ToggleEvent(const AtomString& type, const ToggleEvent::Init& initializer, Event::IsCancelable cancelable)
    : Event(EventInterfaceType::ToggleEvent, type, Event::CanBubble::No, cancelable, Event::IsComposed::No)
    , m_oldState(initializer.oldState)
    , m_newState(initializer.newState)
{
}

ToggleEvent::ToggleEvent(const AtomString& type, const ToggleEvent::Init& initializer)
    : Event(EventInterfaceType::ToggleEvent, type, initializer, IsTrusted::No)
    , m_oldState(initializer.oldState)
    , m_newState(initializer.newState)
{
}

Ref<ToggleEvent> ToggleEvent::create(const AtomString& eventType, const ToggleEvent::Init& init, Event::IsCancelable cancelable)
{
    return adoptRef(*new ToggleEvent(eventType, init, cancelable));
}

Ref<ToggleEvent> ToggleEvent::create(const AtomString& eventType, const ToggleEvent::Init& init)
{
    return adoptRef(*new ToggleEvent(eventType, init));
}

Ref<ToggleEvent> ToggleEvent::createForBindings()
{
    return adoptRef(*new ToggleEvent);
}

} // namespace WebCore
