/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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
#include "CSSBorderImageWidthValue.h"

#include <wtf/text/WTFString.h>

namespace WebCore {

CSSBorderImageWidthValue::CSSBorderImageWidthValue(Quad widths, bool overridesBorderWidths)
    : CSSValue(ClassType::BorderImageWidth)
    , m_widths(WTFMove(widths))
    , m_overridesBorderWidths(overridesBorderWidths)
{
}

CSSBorderImageWidthValue::~CSSBorderImageWidthValue() = default;

Ref<CSSBorderImageWidthValue> CSSBorderImageWidthValue::create(Quad widths, bool overridesBorderWidths)
{
    return adoptRef(*new CSSBorderImageWidthValue(WTFMove(widths), overridesBorderWidths));
}

String CSSBorderImageWidthValue::customCSSText() const
{
    // The border-image-width longhand can't set m_overridesBorderWidths to true, so serialize as empty string.
    // This can only be created by the -webkit-border-image shorthand, which will not serialize as empty string in this case.
    // This is an unconventional relationship between a longhand and a shorthand, which we may want to revise.
    if (m_overridesBorderWidths)
        return String();
    return m_widths.cssText();
}

bool CSSBorderImageWidthValue::equals(const CSSBorderImageWidthValue& other) const
{
    return m_overridesBorderWidths == other.m_overridesBorderWidths && m_widths.equals(other.m_widths);
}

} // namespace WebCore
