/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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
#include "CSSBoxShadowPropertyValue.h"

#include "CSSPrimitiveNumericTypes+CSSValueVisitation.h"
#include "CSSPrimitiveNumericTypes+ComputedStyleDependencies.h"
#include "CSSPrimitiveNumericTypes+Serialization.h"
#include "CSSValuePool.h"
#include "DeprecatedCSSOMBoxShadowValue.h"
#include "DeprecatedCSSOMPrimitiveValue.h"
#include "DeprecatedCSSOMValueList.h"

namespace WebCore {

Ref<CSSBoxShadowPropertyValue> CSSBoxShadowPropertyValue::create(CSS::BoxShadowProperty shadow)
{
    return adoptRef(*new CSSBoxShadowPropertyValue(WTFMove(shadow)));
}

CSSBoxShadowPropertyValue::CSSBoxShadowPropertyValue(CSS::BoxShadowProperty&& shadow)
    : CSSValue(ClassType::BoxShadowProperty)
    , m_shadow(WTFMove(shadow))
{
}

String CSSBoxShadowPropertyValue::customCSSText() const
{
    return CSS::serializationForCSS(m_shadow);
}

bool CSSBoxShadowPropertyValue::equals(const CSSBoxShadowPropertyValue& other) const
{
    return m_shadow == other.m_shadow;
}

IterationStatus CSSBoxShadowPropertyValue::customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
{
    return CSS::visitCSSValueChildren(func, m_shadow);
}

Ref<DeprecatedCSSOMValue> CSSBoxShadowPropertyValue::createDeprecatedCSSOMWrapper(CSSStyleDeclaration& owner) const
{
    return WTF::switchOn(m_shadow,
        [&](CSS::Keyword::None) -> Ref<DeprecatedCSSOMValue> {
            return DeprecatedCSSOMPrimitiveValue::create(CSSPrimitiveValue::create(CSSValueNone), owner);
        },
        [&](const auto& list) -> Ref<DeprecatedCSSOMValue> {
            auto values = list.value.template map<Vector<Ref<DeprecatedCSSOMValue>, 4>>([&](const auto& value) {
                return DeprecatedCSSOMBoxShadowValue::create(value, owner);
            });

            return DeprecatedCSSOMValueList::create(WTFMove(values), CSSValue::CommaSeparator, owner);
        }
    );
}

} // namespace WebCore
