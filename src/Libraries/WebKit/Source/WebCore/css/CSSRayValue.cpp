/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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
#include "CSSRayValue.h"

#include "CSSPrimitiveNumericTypes+CSSValueVisitation.h"
#include "CSSPrimitiveNumericTypes+Serialization.h"
#include "CSSPrimitiveValueMappings.h"
#include "CSSValueKeywords.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

String CSSRayValue::customCSSText() const
{
    StringBuilder builder;
    CSS::serializationForCSS(builder, m_ray);

    if (m_coordinateBox != CSSBoxType::BoxMissing)
        builder.append(' ', nameLiteralForSerialization(toCSSValueID(m_coordinateBox)));

    return builder.toString();
}

bool CSSRayValue::equals(const CSSRayValue& other) const
{
    return m_ray == other.m_ray
        && m_coordinateBox == other.m_coordinateBox;
}

IterationStatus CSSRayValue::customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
{
    return CSS::visitCSSValueChildren(func, m_ray);
}

} // namespace WebCore
