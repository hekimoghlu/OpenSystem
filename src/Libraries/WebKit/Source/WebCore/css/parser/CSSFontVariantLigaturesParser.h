/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"
#include "CSSValuePool.h"

namespace WebCore {

class CSSFontVariantLigaturesParser {
public:
    enum class ParseResult : uint8_t { ConsumedValue, DisallowedValue, UnknownValue };

    CSSFontVariantLigaturesParser() = default;

    ParseResult consumeLigature(CSSParserTokenRange& range)
    {
        CSSValueID valueID = range.peek().id();
        switch (valueID) {
        case CSSValueNoCommonLigatures:
        case CSSValueCommonLigatures:
            if (m_sawCommonLigaturesValue)
                return ParseResult::DisallowedValue;
            m_sawCommonLigaturesValue = true;
            break;
        case CSSValueNoDiscretionaryLigatures:
        case CSSValueDiscretionaryLigatures:
            if (m_sawDiscretionaryLigaturesValue)
                return ParseResult::DisallowedValue;
            m_sawDiscretionaryLigaturesValue = true;
            break;
        case CSSValueNoHistoricalLigatures:
        case CSSValueHistoricalLigatures:
            if (m_sawHistoricalLigaturesValue)
                return ParseResult::DisallowedValue;
            m_sawHistoricalLigaturesValue = true;
            break;
        case CSSValueNoContextual:
        case CSSValueContextual:
            if (m_sawContextualLigaturesValue)
                return ParseResult::DisallowedValue;
            m_sawContextualLigaturesValue = true;
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
    bool m_sawCommonLigaturesValue = false;
    bool m_sawDiscretionaryLigaturesValue = false;
    bool m_sawHistoricalLigaturesValue = false;
    bool m_sawContextualLigaturesValue = false;
    CSSValueListBuilder m_result;
};

} // namespace WebCore
