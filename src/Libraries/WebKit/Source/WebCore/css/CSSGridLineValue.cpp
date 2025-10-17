/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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
#include "CSSGridLineValue.h"

#include <wtf/Vector.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

String CSSGridLineValue::customCSSText() const
{
    Vector<String> parts;
    if (m_spanValue)
        parts.append(m_spanValue->cssText());
    // Only return the numeric value if not 1, or if it provided without a span value.
    // https://drafts.csswg.org/css-grid-2/#grid-placement-span-int
    if (m_numericValue) {
        if (m_numericValue->isOne() != true || !m_spanValue || !m_gridLineName)
            parts.append(m_numericValue->cssText());
    }
    if (m_gridLineName)
        parts.append(m_gridLineName->cssText());
    return makeStringByJoining(parts, " "_s);
}

CSSGridLineValue::CSSGridLineValue(RefPtr<CSSPrimitiveValue>&& spanValue, RefPtr<CSSPrimitiveValue>&& numericValue, RefPtr<CSSPrimitiveValue>&& gridLineName)
    : CSSValue(ClassType::GridLineValue)
    , m_spanValue(WTFMove(spanValue))
    , m_numericValue(WTFMove(numericValue))
    , m_gridLineName(WTFMove(gridLineName))
{
}

Ref<CSSGridLineValue> CSSGridLineValue::create(RefPtr<CSSPrimitiveValue>&& spanValue, RefPtr<CSSPrimitiveValue>&& numericValue, RefPtr<CSSPrimitiveValue>&& gridLineName)
{
    return adoptRef(*new CSSGridLineValue(WTFMove(spanValue), WTFMove(numericValue), WTFMove(gridLineName)));
}

bool CSSGridLineValue::equals(const CSSGridLineValue& other) const
{
    if (m_spanValue) {
        if (!other.m_spanValue || !m_spanValue->equals(*other.m_spanValue))
            return false;
    } else if (other.m_spanValue)
        return false;

    if (m_numericValue) {
        if (!other.m_numericValue || !m_numericValue->equals(*other.m_numericValue))
            return false;
    } else if (other.m_numericValue)
        return false;

    if (m_gridLineName) {
        if (!other.m_gridLineName || !m_gridLineName->equals(*other.m_gridLineName))
            return false;
    } else if (other.m_gridLineName)
        return false;

    return true;
}

} // namespace WebCore
