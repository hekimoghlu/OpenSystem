/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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

#include "LayoutUnit.h"
#include "Length.h"

namespace WebCore {

class FloatSize;
class FloatPoint;
class LayoutSize;

struct Length;
struct LengthSize;
struct LengthPoint;

int intValueForLength(const Length&, LayoutUnit maximumValue);
float floatValueForLength(const Length&, LayoutUnit maximumValue);
WEBCORE_EXPORT float floatValueForLength(const Length&, float maximumValue);
WEBCORE_EXPORT LayoutUnit valueForLength(const Length&, LayoutUnit maximumValue);

LayoutSize sizeForLengthSize(const LengthSize&, const LayoutSize& maximumValue);
FloatSize floatSizeForLengthSize(const LengthSize&, const FloatSize& maximumValue);

LayoutPoint pointForLengthPoint(const LengthPoint&, const LayoutSize& maximumValue);
FloatPoint floatPointForLengthPoint(const LengthPoint&, const FloatSize& maximumValue);

inline LayoutUnit minimumValueForLength(const Length& length, LayoutUnit maximumValue)
{
    switch (length.type()) {
    case LengthType::Fixed:
        return LayoutUnit(length.value());
    case LengthType::Percent:
        // Don't remove the extra cast to float. It is needed for rounding on 32-bit Intel machines that use the FPU stack.
        return LayoutUnit(static_cast<float>(maximumValue * length.percent() / 100.0f));
    case LengthType::Calculated:
        return LayoutUnit(length.nonNanCalculatedValue(maximumValue));
    case LengthType::FillAvailable:
    case LengthType::Auto:
    case LengthType::Normal:
    case LengthType::Content:
        return 0;
    case LengthType::Relative:
    case LengthType::Intrinsic:
    case LengthType::MinIntrinsic:
    case LengthType::MinContent:
    case LengthType::MaxContent:
    case LengthType::FitContent:
    case LengthType::Undefined:
        break;
    }
    ASSERT_NOT_REACHED();
    return 0;
}

inline int minimumIntValueForLength(const Length& length, LayoutUnit maximumValue)
{
    return static_cast<int>(minimumValueForLength(length, maximumValue));
}

template<typename T> inline LayoutUnit valueForLength(const Length& length, T maximumValue) { return valueForLength(length, LayoutUnit(maximumValue)); }
template<typename T> inline LayoutUnit minimumValueForLength(const Length& length, T maximumValue) { return minimumValueForLength(length, LayoutUnit(maximumValue)); }

} // namespace WebCore
