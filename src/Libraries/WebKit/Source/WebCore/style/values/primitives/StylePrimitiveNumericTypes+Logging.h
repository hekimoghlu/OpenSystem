/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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

#include "CSSPrimitiveNumericTypes+Serialization.h"
#include "StylePrimitiveNumericTypes.h"
#include <wtf/text/TextStream.h>

namespace WebCore {
namespace Style {

WTF::TextStream& operator<<(WTF::TextStream& ts, Calc auto const& value)
{
    return ts << value.get();
}

WTF::TextStream& operator<<(WTF::TextStream& ts, Numeric auto const& value)
{
    return ts << CSS::serializationForCSS(CSS::SerializableNumber { value.value, CSS::unitString(value.unit) });
}

WTF::TextStream& operator<<(WTF::TextStream& ts, DimensionPercentageNumeric auto const& value)
{
    return WTF::switchOn(value, [&](const auto& value) -> WTF::TextStream& { return ts << value; });
}

template<auto nR, auto pR, typename V> WTF::TextStream& operator<<(WTF::TextStream& ts, const NumberOrPercentage<nR, pR, V>& value)
{
    return WTF::switchOn(value, [&](const auto& value) -> WTF::TextStream& { return ts << value; });
}

template<auto nR, auto pR, typename V> WTF::TextStream& operator<<(WTF::TextStream& ts, const NumberOrPercentageResolvedToNumber<nR, pR, V>& value)
{
    return ts << value.value;
}

template<typename T> WTF::TextStream& operator<<(WTF::TextStream& ts, const SpaceSeparatedPoint<T>& value)
{
    return ts << value.x() << " " << value.y();
}

template<typename T> WTF::TextStream& operator<<(WTF::TextStream& ts, const SpaceSeparatedSize<T>& value)
{
    return ts << value.width() << " " << value.height();
}

} // namespace Style
} // namespace WebCore
