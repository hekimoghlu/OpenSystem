/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 13, 2024.
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
#include <wtf/GenericTimeMixin.h>
#include <wtf/Int128.h>

namespace WTF {

class MonotonicTime;
class PrintStream;

// The current time according to a wall clock (aka real time clock). This uses floating point
// internally so that you can reason about infinity and other things that arise in math. It's
// acceptable to use this to wrap NaN times, negative times, and infinite times, so long as they
// are relative to the same clock. Use this only if wall clock time is needed. For elapsed time
// measurement use MonotonicTime instead.
class WallTime final : public GenericTimeMixin<WallTime> {
public:
    static constexpr ClockType clockType = ClockType::Wall;
    
    // This is the epoch. So, x.secondsSinceEpoch() should be the same as x - WallTime().
    constexpr WallTime() = default;

    WTF_EXPORT_PRIVATE static WallTime now();
    
    WallTime approximateWallTime() const { return *this; }
    WTF_EXPORT_PRIVATE MonotonicTime approximateMonotonicTime() const;
    
    WTF_EXPORT_PRIVATE void dump(PrintStream&) const;
    
    struct MarkableTraits;

private:
    friend class GenericTimeMixin<WallTime>;
    constexpr WallTime(double rawValue)
        : GenericTimeMixin<WallTime>(rawValue)
    {
    }
};
static_assert(sizeof(WallTime) == sizeof(double));

struct WallTime::MarkableTraits {
    static bool isEmptyValue(WallTime time)
    {
        return time.isNaN();
    }

    static constexpr WallTime emptyValue()
    {
        return WallTime::nan();
    }
};

WTF_EXPORT_PRIVATE Int128 currentTimeInNanoseconds();

} // namespace WTF

using WTF::WallTime;
