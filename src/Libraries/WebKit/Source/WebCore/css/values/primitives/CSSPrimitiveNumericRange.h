/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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

#include <algorithm>
#include <limits>

namespace WebCore {
namespace CSS {

// Options to indicate how the range should be interpreted.
enum class RangeOptions {
    // `Default` indicates that at parse time, out of range values invalidate the parse.
    // Out of range values at style building always clamp.
    Default,

    // `ClampLower` indicates that parse time, an out of range lower value should clamp
    // instead of invalidating the parse. An out of range upper value will still invalidate
    // the parse. Out of range values at style building always clamp.
    ClampLower,

    // `ClampUpper` indicates that parse time, an out of range upper value should clamp
    // instead of invalidating the parse. An out of range lower value will still invalidate
    // the parse. Out of range values at style building always clamp.
    ClampUpper,

    // `ClampBoth` indicates that parse time, an out of range lower or upper value should
    // clamp instead of invalidating the parse. Out of range values at style building
    // always clamp.
    ClampBoth
};

// Representation for `CSS bracketed range notation`. Represents a closed range between (and including) `min` and `max`.
// https://drafts.csswg.org/css-values-4/#numeric-ranges
struct Range {
    // Convenience to allow for a shorter spelling of the appropriate infinity.
    static constexpr auto infinity = std::numeric_limits<double>::infinity();

    double min { -infinity };
    double max {  infinity };
    RangeOptions options { RangeOptions::Default };

    constexpr Range(double min, double max, RangeOptions options = RangeOptions::Default)
        : min { min }
        , max { max }
        , options { options }
    {
    }

    constexpr bool operator==(const Range&) const = default;
};

// Constant value for `[âˆ’âˆž,âˆž]`.
inline constexpr auto All = Range { -Range::infinity, Range::infinity, RangeOptions::Default };

// Constant value for `[0,âˆž]`.
inline constexpr auto Nonnegative = Range { 0, Range::infinity, RangeOptions::Default };

// Constant value for `[0,1]`.
inline constexpr auto ClosedUnitRange = Range { 0, 1 };

// Constant value for `[0,1(clamp upper)]`.
inline constexpr auto ClosedUnitRangeClampUpper = Range { 0, 1, RangeOptions::ClampUpper };

// Constant value for `[0,100]`.
inline constexpr auto ClosedPercentageRange = Range { 0, 100 };

// Constant value for `[0,100(clamp upper)]`.
inline constexpr auto ClosedPercentageRangeClampUpper = Range { 0, 100, RangeOptions::ClampUpper };

// Clamps a floating point value to within `range`.
template<Range range, std::floating_point T> constexpr T clampToRange(T value)
{
    return std::clamp<T>(value, range.min, range.max);
}

} // namespace CSS
} // namespace WebCore
