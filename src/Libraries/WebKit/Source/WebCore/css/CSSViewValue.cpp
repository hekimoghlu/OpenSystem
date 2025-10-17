/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
#include "CSSViewValue.h"

#include "CSSPrimitiveValueMappings.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

String CSSViewValue::customCSSText() const
{
    auto hasAxis = m_axis && m_axis->valueID() != CSSValueBlock;
    auto hasEndInset = m_endInset && m_endInset != m_startInset;
    auto hasStartInset = (m_startInset && m_startInset->valueID() != CSSValueAuto) || (m_startInset && m_startInset->valueID() == CSSValueAuto && hasEndInset);

    return makeString(
        "view("_s,
        hasAxis ? m_axis->cssText() : ""_s,
        hasAxis && hasStartInset ? " "_s : ""_s,
        hasStartInset ? m_startInset->cssText() : ""_s,
        hasStartInset && hasEndInset ? " "_s : ""_s,
        hasEndInset ? m_endInset->cssText() : ""_s,
        ")"_s
    );
}

bool CSSViewValue::equals(const CSSViewValue& other) const
{
    return compareCSSValuePtr(m_axis, other.m_axis)
        && compareCSSValuePtr(m_startInset, other.m_startInset)
        && compareCSSValuePtr(m_endInset, other.m_endInset);
}

}
