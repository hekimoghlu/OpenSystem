/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#include "CSSBackgroundRepeatValue.h"

#include "CSSValueKeywords.h"
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

CSSBackgroundRepeatValue::CSSBackgroundRepeatValue(CSSValueID xValue, CSSValueID yValue)
    : CSSValue(ClassType::BackgroundRepeat)
    , m_xValue(xValue)
    , m_yValue(yValue)
{
}

Ref<CSSBackgroundRepeatValue> CSSBackgroundRepeatValue::create(CSSValueID repeatXValue, CSSValueID repeatYValue)
{
    return adoptRef(*new CSSBackgroundRepeatValue(repeatXValue, repeatYValue));
}

String CSSBackgroundRepeatValue::customCSSText() const
{
    // background-repeat/mask-repeat behave a little like a shorthand, but `repeat no-repeat` is transformed to `repeat-x`.
    if (m_xValue != m_yValue) {
        if (m_xValue == CSSValueRepeat && m_yValue == CSSValueNoRepeat)
            return nameString(CSSValueRepeatX);
        if (m_xValue == CSSValueNoRepeat && m_yValue == CSSValueRepeat)
            return nameString(CSSValueRepeatY);
        return makeString(nameLiteral(m_xValue), ' ', nameLiteral(m_yValue));
    }
    return nameString(m_xValue);
}

bool CSSBackgroundRepeatValue::equals(const CSSBackgroundRepeatValue& other) const
{
    return m_xValue == other.m_xValue && m_yValue == other.m_yValue;
}

} // namespace WebCore
