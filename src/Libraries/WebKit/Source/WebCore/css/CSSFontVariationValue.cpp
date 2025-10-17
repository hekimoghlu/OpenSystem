/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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
#include "CSSFontVariationValue.h"

#include "CSSPrimitiveValue.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

CSSFontVariationValue::CSSFontVariationValue(FontTag tag, Ref<CSSPrimitiveValue>&& value)
    : CSSValue(ClassType::FontVariation)
    , m_tag(tag)
    , m_value(WTFMove(value))
{
}

String CSSFontVariationValue::customCSSText() const
{
    return makeString('"', m_tag[0], m_tag[1], m_tag[2], m_tag[3], "\" "_s, m_value->customCSSText());
}

bool CSSFontVariationValue::equals(const CSSFontVariationValue& other) const
{
    return m_tag == other.m_tag && compareCSSValue(m_value, other.m_value);
}

}
