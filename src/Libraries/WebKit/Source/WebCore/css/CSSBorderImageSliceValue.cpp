/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
#include "CSSBorderImageSliceValue.h"

#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

CSSBorderImageSliceValue::CSSBorderImageSliceValue(Quad slices, bool fill)
    : CSSValue(ClassType::BorderImageSlice)
    , m_slices(WTFMove(slices))
    , m_fill(fill)
{
}

CSSBorderImageSliceValue::~CSSBorderImageSliceValue() = default;

Ref<CSSBorderImageSliceValue> CSSBorderImageSliceValue::create(Quad slices, bool fill)
{
    return adoptRef(*new CSSBorderImageSliceValue(WTFMove(slices), fill));
}

String CSSBorderImageSliceValue::customCSSText() const
{
    if (m_fill)
        return makeString(m_slices.cssText(), " fill"_s);
    return m_slices.cssText();
}

bool CSSBorderImageSliceValue::equals(const CSSBorderImageSliceValue& other) const
{
    return m_fill == other.m_fill && m_slices.equals(other.m_slices);
}

} // namespace WebCore
