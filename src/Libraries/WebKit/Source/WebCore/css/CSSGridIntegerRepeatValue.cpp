/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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
#include "CSSGridIntegerRepeatValue.h"

#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSGridIntegerRepeatValue::CSSGridIntegerRepeatValue(Ref<CSSPrimitiveValue>&& repetitions, CSSValueListBuilder builder)
    : CSSValueContainingVector(ClassType::GridIntegerRepeat, SpaceSeparator, WTFMove(builder))
    , m_repetitions(WTFMove(repetitions))
{
}

Ref<CSSGridIntegerRepeatValue> CSSGridIntegerRepeatValue::create(Ref<CSSPrimitiveValue>&& repetitions, CSSValueListBuilder builder)
{
    return adoptRef(*new CSSGridIntegerRepeatValue(WTFMove(repetitions), WTFMove(builder)));
}

String CSSGridIntegerRepeatValue::customCSSText() const
{
    StringBuilder result;
    result.append("repeat("_s, m_repetitions->cssText(), ", "_s);
    serializeItems(result);
    result.append(')');
    return result.toString();
}

bool CSSGridIntegerRepeatValue::equals(const CSSGridIntegerRepeatValue& other) const
{
    return compareCSSValue(m_repetitions, other.m_repetitions) && itemsEqual(other);
}

} // namespace WebCore
