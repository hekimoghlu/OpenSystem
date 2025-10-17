/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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

namespace WTF {

class WallTime;
class PrintStream;

// The current time according to an approximate monotonic clock. Similar to MonotonicTime, but its resolution is
// coarse, instead ApproximateTime::now() is much faster than MonotonicTime::now().
class ApproximateTime final : public GenericTimeMixin<ApproximateTime> {
public:
    static constexpr ClockType clockType = ClockType::Approximate;

    // This is the epoch. So, x.secondsSinceEpoch() should be the same as x - ApproximateTime().
    constexpr ApproximateTime() = default;

#if OS(DARWIN)
    WTF_EXPORT_PRIVATE static ApproximateTime fromMachApproximateTime(uint64_t);
    WTF_EXPORT_PRIVATE uint64_t toMachApproximateTime() const;
#endif

    WTF_EXPORT_PRIVATE static ApproximateTime now();

    ApproximateTime approximateApproximateTime() const { return *this; }
    WTF_EXPORT_PRIVATE WallTime approximateWallTime() const;
    WTF_EXPORT_PRIVATE MonotonicTime approximateMonotonicTime() const;

    WTF_EXPORT_PRIVATE void dump(PrintStream&) const;

    struct MarkableTraits;

private:
    friend class GenericTimeMixin<ApproximateTime>;
    constexpr ApproximateTime(double rawValue)
        : GenericTimeMixin<ApproximateTime>(rawValue)
    {
    }
};
static_assert(sizeof(ApproximateTime) == sizeof(double));

struct ApproximateTime::MarkableTraits {
    static bool isEmptyValue(ApproximateTime time)
    {
        return time.isNaN();
    }

    static constexpr ApproximateTime emptyValue()
    {
        return ApproximateTime::nan();
    }
};

} // namespace WTF

using WTF::ApproximateTime;
