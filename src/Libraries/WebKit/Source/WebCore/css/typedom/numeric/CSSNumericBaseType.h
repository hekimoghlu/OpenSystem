/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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

#include <array>
#include <wtf/text/ASCIILiteral.h>

namespace WebCore {

enum class CSSNumericBaseType : uint8_t {
    Length,
    Angle,
    Time,
    Frequency,
    Resolution,
    Flex,
    Percent,
};

constexpr std::array<CSSNumericBaseType, 7> eachBaseType()
{
    return {
        CSSNumericBaseType::Length,
        CSSNumericBaseType::Angle,
        CSSNumericBaseType::Time,
        CSSNumericBaseType::Frequency,
        CSSNumericBaseType::Resolution,
        CSSNumericBaseType::Flex,
        CSSNumericBaseType::Percent
    };
}

constexpr ASCIILiteral debugString(CSSNumericBaseType type)
{
    switch (type) {
    case CSSNumericBaseType::Length:
        return "length"_s;
    case CSSNumericBaseType::Angle:
        return "angle"_s;
    case CSSNumericBaseType::Time:
        return "time"_s;
    case CSSNumericBaseType::Frequency:
        return "frequency"_s;
    case CSSNumericBaseType::Resolution:
        return "resolution"_s;
    case CSSNumericBaseType::Flex:
        return "flex"_s;
    case CSSNumericBaseType::Percent:
        return "percent"_s;
    }
    return "invalid"_s;
}

} // namespace WebCore
