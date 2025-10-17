/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 15, 2023.
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
#include "CSSColorValue.h"

#include "CSSPrimitiveValue.h"

namespace WebCore {

Ref<CSSColorValue> CSSColorValue::create(CSS::Color color)
{
    return adoptRef(*new CSSColorValue(WTFMove(color)));
}

Ref<CSSColorValue> CSSColorValue::create(WebCore::Color color)
{
    return adoptRef(*new CSSColorValue(WTFMove(color)));
}

CSSColorValue::CSSColorValue(CSS::Color color)
    : CSSValue(ClassType::Color)
    , m_color(WTFMove(color))
{
}

CSSColorValue::CSSColorValue(WebCore::Color color)
    : CSSColorValue(CSS::Color { CSS::ResolvedColor { WTFMove(color) } })
{
}

CSSColorValue::CSSColorValue(StaticCSSValueTag, WebCore::Color color)
    : CSSColorValue(WTFMove(color))
{
    makeStatic();
}

WebCore::Color CSSColorValue::absoluteColor(const CSSValue& value)
{
    if (RefPtr color = dynamicDowncast<CSSColorValue>(value))
        return color->color().absoluteColor();

    if (auto valueID = value.valueID(); CSS::isAbsoluteColorKeyword(valueID))
        return CSS::colorFromAbsoluteKeyword(valueID);
    return { };
}

String CSSColorValue::customCSSText() const
{
    return CSS::serializationForCSS(m_color);
}

bool CSSColorValue::equals(const CSSColorValue& other) const
{
    return m_color == other.m_color;
}

IterationStatus CSSColorValue::customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
{
    return CSS::visitCSSValueChildren(func, m_color);
}

} // namespace WebCore
