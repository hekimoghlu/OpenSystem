/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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

#include "CSSParserTokenRange.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"
#include "CSSValuePool.h"

namespace WebCore {

class CSSFontVariantNumericParser {
public:
    enum class ParseResult : uint8_t { ConsumedValue, DisallowedValue, UnknownValue };

    CSSFontVariantNumericParser() = default;

    ParseResult consumeNumeric(CSSParserTokenRange& range)
    {
        CSSValueID valueID = range.peek().id();
        switch (valueID) {
        case CSSValueLiningNums:
        case CSSValueOldstyleNums:
            if (m_sawNumericFigureValue)
                return ParseResult::DisallowedValue;
            m_sawNumericFigureValue = true;
            break;
        case CSSValueProportionalNums:
        case CSSValueTabularNums:
            if (m_sawNumericSpacingValue)
                return ParseResult::DisallowedValue;
            m_sawNumericSpacingValue = true;
            break;
        case CSSValueDiagonalFractions:
        case CSSValueStackedFractions:
            if (m_sawNumericFractionValue)
                return ParseResult::DisallowedValue;
            m_sawNumericFractionValue = true;
            break;
        case CSSValueOrdinal:
            if (m_sawOrdinalValue)
                return ParseResult::DisallowedValue;
            m_sawOrdinalValue = true;
            break;
        case CSSValueSlashedZero:
            if (m_sawSlashedZeroValue)
                return ParseResult::DisallowedValue;
            m_sawSlashedZeroValue = true;
            break;
        default:
            return ParseResult::UnknownValue;
        }
        m_result.append(CSSPropertyParserHelpers::consumeIdent(range).releaseNonNull());
        return ParseResult::ConsumedValue;
    }

    RefPtr<CSSValue> finalizeValue()
    {
        if (m_result.isEmpty())
            return CSSPrimitiveValue::create(CSSValueNormal);
        return CSSValueList::createSpaceSeparated(WTFMove(m_result));
    }

private:
    bool m_sawNumericFigureValue : 1 { false };
    bool m_sawNumericSpacingValue : 1 { false };
    bool m_sawNumericFractionValue : 1 { false };
    bool m_sawOrdinalValue : 1 { false };
    bool m_sawSlashedZeroValue : 1 { false };
    CSSValueListBuilder m_result;
};

} // namespace WebCore
