/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
#include "CSSGridAutoRepeatValue.h"

#include "CSSValueKeywords.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSGridAutoRepeatValue::CSSGridAutoRepeatValue(bool isAutoFit, CSSValueListBuilder builder)
    : CSSValueContainingVector(ClassType::GridAutoRepeat, SpaceSeparator, WTFMove(builder))
    , m_isAutoFit(isAutoFit)
{
}

Ref<CSSGridAutoRepeatValue> CSSGridAutoRepeatValue::create(CSSValueID id, CSSValueListBuilder builder)
{
    ASSERT(id == CSSValueAutoFill || id == CSSValueAutoFit);
    return adoptRef(*new CSSGridAutoRepeatValue(id == CSSValueAutoFit, WTFMove(builder)));
}

CSSValueID CSSGridAutoRepeatValue::autoRepeatID() const
{
    return m_isAutoFit ? CSSValueAutoFit : CSSValueAutoFill;
}

String CSSGridAutoRepeatValue::customCSSText() const
{
    StringBuilder result;
    result.append("repeat("_s, nameLiteral(autoRepeatID()), ", "_s);
    serializeItems(result);
    result.append(')');
    return result.toString();
}

bool CSSGridAutoRepeatValue::equals(const CSSGridAutoRepeatValue& other) const
{
    return m_isAutoFit == other.m_isAutoFit && itemsEqual(other);
}

} // namespace WebCore
