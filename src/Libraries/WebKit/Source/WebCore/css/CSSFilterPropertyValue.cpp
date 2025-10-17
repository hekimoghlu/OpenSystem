/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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
#include "CSSFilterPropertyValue.h"

#include "CSSPrimitiveNumericTypes+CSSValueVisitation.h"
#include "CSSPrimitiveNumericTypes+Serialization.h"
#include "CSSValuePool.h"
#include "DeprecatedCSSOMFilterFunctionValue.h"
#include "DeprecatedCSSOMPrimitiveValue.h"
#include "DeprecatedCSSOMValueList.h"

namespace WebCore {

Ref<CSSFilterPropertyValue> CSSFilterPropertyValue::create(CSS::FilterProperty filter)
{
    return adoptRef(*new CSSFilterPropertyValue(WTFMove(filter)));
}

CSSFilterPropertyValue::CSSFilterPropertyValue(CSS::FilterProperty filter)
    : CSSValue(ClassType::FilterProperty)
    , m_filter(WTFMove(filter))
{
}

String CSSFilterPropertyValue::customCSSText() const
{
    return CSS::serializationForCSS(m_filter);
}

bool CSSFilterPropertyValue::equals(const CSSFilterPropertyValue& other) const
{
    return m_filter == other.m_filter;
}

IterationStatus CSSFilterPropertyValue::customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
{
    return CSS::visitCSSValueChildren(func, m_filter);
}

Ref<DeprecatedCSSOMValue> CSSFilterPropertyValue::createDeprecatedCSSOMWrapper(CSSStyleDeclaration& owner) const
{
    return WTF::switchOn(m_filter,
        [&](CSS::Keyword::None) -> Ref<DeprecatedCSSOMValue> {
            return DeprecatedCSSOMPrimitiveValue::create(CSSPrimitiveValue::create(CSSValueNone), owner);
        },
        [&](const auto& list) -> Ref<DeprecatedCSSOMValue> {
            auto values = list.value.template map<Vector<Ref<DeprecatedCSSOMValue>, 4>>([&](const auto& value) {
                return WTF::switchOn(value,
                    [&](const CSS::FilterReference& reference) -> Ref<DeprecatedCSSOMValue> {
                        return DeprecatedCSSOMPrimitiveValue::create(CSSPrimitiveValue::createURI(reference.url), owner);
                    },
                    [&](const auto& function) -> Ref<DeprecatedCSSOMValue> {
                        return DeprecatedCSSOMFilterFunctionValue::create(CSS::FilterFunction { function }, owner);
                    }
                );
            });

            return DeprecatedCSSOMValueList::create(WTFMove(values), CSSValue::SpaceSeparator, owner);
        }
    );
}

} // namespace WebCore
