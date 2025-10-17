/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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
#include "CSSReflectValue.h"

#include <wtf/text/MakeString.h>

namespace WebCore {

CSSReflectValue::CSSReflectValue(CSSValueID direction, Ref<CSSPrimitiveValue> offset, RefPtr<CSSValue> mask)
    : CSSValue(ClassType::Reflect)
    , m_direction(direction)
    , m_offset(WTFMove(offset))
    , m_mask(WTFMove(mask))
{
}

Ref<CSSReflectValue> CSSReflectValue::create(CSSValueID direction, Ref<CSSPrimitiveValue> offset, RefPtr<CSSValue> mask)
{
    return adoptRef(*new CSSReflectValue(direction, WTFMove(offset), WTFMove(mask)));
}

String CSSReflectValue::customCSSText() const
{
    if (m_mask)
        return makeString(nameLiteral(m_direction), ' ', m_offset->cssText(), ' ', m_mask->cssText());
    return makeString(nameLiteral(m_direction), ' ', m_offset->cssText());
}

bool CSSReflectValue::equals(const CSSReflectValue& other) const
{
    return m_direction == other.m_direction
        && compareCSSValue(m_offset, other.m_offset)
        && compareCSSValuePtr(m_mask, other.m_mask);
}

} // namespace WebCore
