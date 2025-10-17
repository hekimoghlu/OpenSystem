/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 20, 2022.
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
#include "CSSColorSchemeValue.h"

#if ENABLE(DARK_MODE_CSS)

namespace WebCore {

Ref<CSSColorSchemeValue> CSSColorSchemeValue::create(CSS::ColorScheme colorScheme)
{
    return adoptRef(*new CSSColorSchemeValue(WTFMove(colorScheme)));
}

CSSColorSchemeValue::CSSColorSchemeValue(CSS::ColorScheme colorScheme)
    : CSSValue(ClassType::ColorScheme)
    , m_colorScheme(WTFMove(colorScheme))
{
}

String CSSColorSchemeValue::customCSSText() const
{
    return CSS::serializationForCSS(m_colorScheme);
}

bool CSSColorSchemeValue::equals(const CSSColorSchemeValue& other) const
{
    return m_colorScheme == other.m_colorScheme;
}

IterationStatus CSSColorSchemeValue::customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
{
    return CSS::visitCSSValueChildren(func, m_colorScheme);
}

} // namespace WebCore

#endif
