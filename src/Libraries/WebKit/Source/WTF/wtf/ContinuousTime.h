/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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

// The current time according to a continuous monotonic clock, which continues ticking while the
// system is asleep.
class ContinuousTime final : public GenericTimeMixin<ContinuousTime> {
public:
    static constexpr ClockType clockType = ClockType::Continuous;

    // This is the epoch. So, x.secondsSinceEpoch() should be the same as x - ContinuousTime().
    constexpr ContinuousTime() = default;

#if OS(DARWIN)
    WTF_EXPORT_PRIVATE static ContinuousTime fromMachContinuousTime(uint64_t);
    WTF_EXPORT_PRIVATE uint64_t toMachContinuousTime() const;
#endif

    WTF_EXPORT_PRIVATE static ContinuousTime now();

    WTF_EXPORT_PRIVATE WallTime approximateWallTime() const;
    WTF_EXPORT_PRIVATE MonotonicTime approximateMonotonicTime() const;

    WTF_EXPORT_PRIVATE void dump(PrintStream&) const;

    struct MarkableTraits;

private:
    friend class GenericTimeMixin<ContinuousTime>;
    constexpr ContinuousTime(double rawValue)
        : GenericTimeMixin<ContinuousTime>(rawValue)
    {
    }
};
static_assert(sizeof(ContinuousTime) == sizeof(double));

struct ContinuousTime::MarkableTraits {
    static bool isEmptyValue(ContinuousTime time)
    {
        return time.isNaN();
    }

    static constexpr ContinuousTime emptyValue()
    {
        return ContinuousTime::nan();
    }
};

} // namespace WTF

using WTF::ContinuousTime;
