/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
#include "LengthFunctions.h"

#include "FloatSize.h"
#include "LayoutSize.h"
#include "LengthPoint.h"
#include "LengthSize.h"

namespace WebCore {

int intValueForLength(const Length& length, LayoutUnit maximumValue)
{
    return static_cast<int>(valueForLength(length, maximumValue));
}

LayoutUnit valueForLength(const Length& length, LayoutUnit maximumValue)
{
    switch (length.type()) {
    case LengthType::Fixed:
    case LengthType::Percent:
    case LengthType::Calculated:
        return minimumValueForLength(length, maximumValue);
    case LengthType::FillAvailable:
    case LengthType::Auto:
    case LengthType::Normal:
        return maximumValue;
    case LengthType::Relative:
    case LengthType::Intrinsic:
    case LengthType::MinIntrinsic:
    case LengthType::Content:
    case LengthType::MinContent:
    case LengthType::MaxContent:
    case LengthType::FitContent:
    case LengthType::Undefined:
        ASSERT_NOT_REACHED();
        return 0;
    }
    ASSERT_NOT_REACHED();
    return 0;
}

// FIXME: when subpixel layout is supported this copy of floatValueForLength() can be removed. See bug 71143.
float floatValueForLength(const Length& length, LayoutUnit maximumValue)
{
    switch (length.type()) {
    case LengthType::Fixed:
        return length.value();
    case LengthType::Percent:
        return static_cast<float>(maximumValue * length.percent() / 100.0f);
    case LengthType::FillAvailable:
    case LengthType::Auto:
    case LengthType::Normal:
        return static_cast<float>(maximumValue);
    case LengthType::Calculated:
        return length.nonNanCalculatedValue(maximumValue);
    case LengthType::Relative:
    case LengthType::Intrinsic:
    case LengthType::MinIntrinsic:
    case LengthType::Content:
    case LengthType::MinContent:
    case LengthType::MaxContent:
    case LengthType::FitContent:
    case LengthType::Undefined:
        ASSERT_NOT_REACHED();
        return 0;
    }
    ASSERT_NOT_REACHED();
    return 0;
}

float floatValueForLength(const Length& length, float maximumValue)
{
    switch (length.type()) {
    case LengthType::Fixed:
        return length.value();
    case LengthType::Percent:
        return static_cast<float>(maximumValue * length.percent() / 100.0f);
    case LengthType::FillAvailable:
    case LengthType::Auto:
    case LengthType::Normal:
        return static_cast<float>(maximumValue);
    case LengthType::Calculated:
        return length.nonNanCalculatedValue(maximumValue);
    case LengthType::Relative:
    case LengthType::Intrinsic:
    case LengthType::MinIntrinsic:
    case LengthType::Content:
    case LengthType::MinContent:
    case LengthType::MaxContent:
    case LengthType::FitContent:
    case LengthType::Undefined:
        ASSERT_NOT_REACHED();
        return 0;
    }
    ASSERT_NOT_REACHED();
    return 0;
}

LayoutSize sizeForLengthSize(const LengthSize& length, const LayoutSize& maximumValue)
{
    return { valueForLength(length.width, maximumValue.width()), valueForLength(length.height, maximumValue.height()) };
}

LayoutPoint pointForLengthPoint(const LengthPoint& lengthPoint, const LayoutSize& maximumValue)
{
    return { valueForLength(lengthPoint.x, maximumValue.width()), valueForLength(lengthPoint.y, maximumValue.height()) };
}

FloatSize floatSizeForLengthSize(const LengthSize& lengthSize, const FloatSize& boxSize)
{
    return { floatValueForLength(lengthSize.width, boxSize.width()), floatValueForLength(lengthSize.height, boxSize.height()) };
}

FloatPoint floatPointForLengthPoint(const LengthPoint& lengthPoint, const FloatSize& boxSize)
{
    return { floatValueForLength(lengthPoint.x, boxSize.width()), floatValueForLength(lengthPoint.y, boxSize.height()) };
}

} // namespace WebCore
