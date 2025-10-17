/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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

#include <wtf/Condition.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/TimeWithDynamicClockType.h>

namespace WTF {

class BinarySemaphore final {
    WTF_MAKE_NONCOPYABLE(BinarySemaphore);
    WTF_MAKE_FAST_ALLOCATED;
public:
    constexpr BinarySemaphore() = default;

    WTF_EXPORT_PRIVATE void signal();
    WTF_EXPORT_PRIVATE bool waitUntil(const TimeWithDynamicClockType&);

    bool waitFor(Seconds relativeTimeout)
    {
        return waitUntil(MonotonicTime::timePointFromNow(relativeTimeout));
    }

    void wait()
    {
        waitUntil(ParkingLot::Time::infinity());
    }

private:
    bool m_isSet WTF_GUARDED_BY_LOCK(m_lock) { false };
    Lock m_lock;
    Condition m_condition;
};

} // namespace WTF

using WTF::BinarySemaphore;
