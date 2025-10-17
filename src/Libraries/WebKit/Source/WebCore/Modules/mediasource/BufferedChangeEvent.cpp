/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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

#if ENABLE(MEDIA_SOURCE)
#include "BufferedChangeEvent.h"

#include "Event.h"
#include "EventNames.h"
#include "TimeRanges.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(BufferedChangeEvent);

BufferedChangeEvent::BufferedChangeEvent(RefPtr<TimeRanges>&& added, RefPtr<TimeRanges>&& removed)
    : Event(EventInterfaceType::BufferedChangeEvent, eventNames().bufferedchangeEvent, CanBubble::No, IsCancelable::No)
    , m_added(WTFMove(added))
    , m_removed(WTFMove(removed))
{
}

BufferedChangeEvent::BufferedChangeEvent(const AtomString& type, Init&& init)
    : Event(EventInterfaceType::BufferedChangeEvent, type, init, IsTrusted::No)
    , m_added(WTFMove(init.addedRanges))
    , m_removed(WTFMove(init.removedRanges))
{
}

BufferedChangeEvent::~BufferedChangeEvent() = default;

RefPtr<TimeRanges> BufferedChangeEvent::addedRanges() const
{
    return m_added;
}

RefPtr<TimeRanges> BufferedChangeEvent::removedRanges() const
{
    return m_removed;
}

} // namespace WebCore

#endif // ENABLE(MEDIA_SOURCE)
