/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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

#include "FloatSize.h"
#include "ScrollTypes.h"
#include <wtf/Deque.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class PlatformWheelEvent;

class WheelEventDeltaFilter {
public:
    WheelEventDeltaFilter();
    virtual ~WheelEventDeltaFilter();

    WEBCORE_EXPORT static std::unique_ptr<WheelEventDeltaFilter> create();

    WEBCORE_EXPORT virtual void updateFromEvent(const PlatformWheelEvent&) = 0;

    WEBCORE_EXPORT PlatformWheelEvent eventCopyWithFilteredDeltas(const PlatformWheelEvent&) const;
    WEBCORE_EXPORT PlatformWheelEvent eventCopyWithVelocity(const PlatformWheelEvent&) const;

    WEBCORE_EXPORT FloatSize filteredVelocity() const;
    WEBCORE_EXPORT FloatSize filteredDelta() const;

    WEBCORE_EXPORT static bool shouldApplyFilteringForEvent(const PlatformWheelEvent&);
    WEBCORE_EXPORT static bool shouldIncludeVelocityForEvent(const PlatformWheelEvent&);

protected:
    FloatSize m_currentFilteredDelta;
    FloatSize m_currentFilteredVelocity;
};

class BasicWheelEventDeltaFilter final : public WheelEventDeltaFilter {
    WTF_MAKE_TZONE_ALLOCATED(BasicWheelEventDeltaFilter);
public:
    BasicWheelEventDeltaFilter();
    void updateFromEvent(const PlatformWheelEvent&) final;

private:
    std::optional<ScrollEventAxis> dominantAxis() const;

    void reset();
    void updateWithDelta(FloatSize);

    Deque<FloatSize> m_recentWheelEventDeltas;
};

} // namespace WebCore
