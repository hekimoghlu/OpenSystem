/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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
#pragma once

#if ENABLE(MEDIA_SOURCE)

#include "Event.h"

namespace WebCore {

class TimeRanges;

class BufferedChangeEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(BufferedChangeEvent);
public:
    ~BufferedChangeEvent();

    struct Init : EventInit {
        RefPtr<TimeRanges> addedRanges;
        RefPtr<TimeRanges> removedRanges;
    };

    static Ref<BufferedChangeEvent> create(RefPtr<TimeRanges>&& added, RefPtr<TimeRanges>&& removed)
    {
        return adoptRef(*new BufferedChangeEvent(WTFMove(added), WTFMove(removed)));
    }

    static Ref<BufferedChangeEvent> create(const AtomString& type, Init&& init)
    {
        return adoptRef(*new BufferedChangeEvent(type, WTFMove(init)));
    }

    RefPtr<TimeRanges> addedRanges() const;
    RefPtr<TimeRanges> removedRanges() const;

private:
    BufferedChangeEvent(RefPtr<TimeRanges>&& added, RefPtr<TimeRanges>&& removed);
    BufferedChangeEvent(const AtomString& type, Init&&);

    RefPtr<TimeRanges> m_added;
    RefPtr<TimeRanges> m_removed;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_SOURCE)
