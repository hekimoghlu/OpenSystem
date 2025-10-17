/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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
#include "CSSAttrValue.h"

#include <wtf/text/MakeString.h>
#include "CSSPrimitiveValue.h"

namespace WebCore {

Ref<CSSAttrValue> CSSAttrValue::create(String attributeName, RefPtr<CSSValue>&& fallback)
{
    return adoptRef(*new CSSAttrValue(WTFMove(attributeName), WTFMove(fallback)));
}

bool CSSAttrValue::equals(const CSSAttrValue& other) const
{
    RefPtr fallback = dynamicDowncast<CSSPrimitiveValue>(m_fallback);
    RefPtr otherFallback = dynamicDowncast<CSSPrimitiveValue>(other.m_fallback);

    if (fallback && otherFallback)
        return m_attributeName == other.m_attributeName && fallback->stringValue() == otherFallback->stringValue();
    if (fallback || otherFallback)
        return false;
    return m_attributeName == other.m_attributeName;
}

String CSSAttrValue::customCSSText() const
{
    RefPtr fallback = dynamicDowncast<CSSPrimitiveValue>(m_fallback);
    return makeString(
        "attr("_s,
        m_attributeName.impl(),
        fallback && !fallback->stringValue().isEmpty() ? ", "_s : ""_s,
        fallback && !fallback->stringValue().isEmpty() ? fallback->cssText() : ""_s,
        ')'
    );
}

} // namespace WebCore
