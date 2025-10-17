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

#include <wtf/ClockType.h>
#include <wtf/Forward.h>
#include <wtf/GenericTimeMixin.h>

namespace WTF {

class WallTime;
class PrintStream;

// The current time according to a monotonic clock. Monotonic clocks don't go backwards and
// possibly don't count downtime. This uses floating point internally so that you can reason about
// infinity and other things that arise in math. It's acceptable to use this to wrap NaN times,
// negative times, and infinite times, so long as they are all relative to the same clock.
class MonotonicTime final : public GenericTimeMixin<MonotonicTime> {
public:
    static constexpr ClockType clockType = ClockType::Monotonic;
    
    // This is the epoch. So, x.secondsSinceEpoch() should be the same as x - MonotonicTime().
    constexpr MonotonicTime() = default;

#if OS(DARWIN)
    WTF_EXPORT_PRIVATE static MonotonicTime fromMachAbsoluteTime(uint64_t);
    WTF_EXPORT_PRIVATE uint64_t toMachAbsoluteTime() const;
#endif

    WTF_EXPORT_PRIVATE static MonotonicTime now();
    
    MonotonicTime approximateMonotonicTime() const { return *this; }
    WTF_EXPORT_PRIVATE WallTime approximateWallTime() const;

    WTF_EXPORT_PRIVATE void dump(PrintStream&) const;

    struct MarkableTraits;

private:
    friend class GenericTimeMixin<MonotonicTime>;
    constexpr MonotonicTime(double rawValue)
        : GenericTimeMixin<MonotonicTime>(rawValue)
    {
    }
};
static_assert(sizeof(MonotonicTime) == sizeof(double));

struct MonotonicTime::MarkableTraits {
    static bool isEmptyValue(MonotonicTime time)
    {
        return std::isnan(time.m_value);
    }

    static constexpr MonotonicTime emptyValue()
    {
        return MonotonicTime::nan();
    }
};

} // namespace WTF
