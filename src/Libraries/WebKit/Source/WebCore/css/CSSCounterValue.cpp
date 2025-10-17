/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 26, 2023.
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
#include "CSSCounterValue.h"

#include "CSSMarkup.h"
#include "CSSPrimitiveValue.h"
#include <wtf/PointerComparison.h> 
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSCounterValue::CSSCounterValue(AtomString identifier, AtomString separator, RefPtr<CSSValue> counterStyle)
    : CSSValue(ClassType::Counter)
    , m_identifier(WTFMove(identifier))
    , m_separator(WTFMove(separator))
    , m_counterStyle(WTFMove(counterStyle))
{
}

Ref<CSSCounterValue> CSSCounterValue::create(AtomString identifier, AtomString separator, RefPtr<CSSValue> counterStyle)
{
    return adoptRef(*new CSSCounterValue(WTFMove(identifier), WTFMove(separator), WTFMove(counterStyle)));
}

bool CSSCounterValue::equals(const CSSCounterValue& other) const
{
    return m_identifier == other.m_identifier && m_separator == other.m_separator && arePointingToEqualData(m_counterStyle, other.m_counterStyle);
}

String CSSCounterValue::customCSSText() const
{
    bool isDecimal = m_counterStyle->valueID() == CSSValueDecimal || (m_counterStyle->isCustomIdent() && m_counterStyle->customIdent() == "decimal"_s);
    auto listStyleSeparator = isDecimal ? ""_s : ", "_s;
    auto listStyleLiteral = isDecimal ? ""_s : counterStyleCSSText();
    if (m_separator.isEmpty())
        return makeString("counter("_s, m_identifier, listStyleSeparator, listStyleLiteral, ')');
    StringBuilder result;
    result.append("counters("_s, m_identifier, ", "_s);
    serializeString(m_separator, result);
    result.append(listStyleSeparator, listStyleLiteral, ')');
    return result.toString();
}

String CSSCounterValue::counterStyleCSSText() const
{
    if (!m_counterStyle)
        return emptyString();

    if (m_counterStyle->isValueID())
        return nameString(m_counterStyle->valueID()).string();
    if (m_counterStyle->isCustomIdent())
        return m_counterStyle->customIdent();

    ASSERT_NOT_REACHED();
    return emptyString();
}
} // namespace WebCore
