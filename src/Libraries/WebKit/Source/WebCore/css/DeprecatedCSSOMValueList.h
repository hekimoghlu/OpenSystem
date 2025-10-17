/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 1, 2025.
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
#pragma once

#include "CSSValueList.h"
#include "DeprecatedCSSOMValue.h"

namespace WebCore {
    
class DeprecatedCSSOMValueList : public DeprecatedCSSOMValue {
public:
    static Ref<DeprecatedCSSOMValueList> create(Vector<Ref<DeprecatedCSSOMValue>, 4> values, CSSValue::ValueSeparator separator, CSSStyleDeclaration& owner)
    {
        return adoptRef(*new DeprecatedCSSOMValueList(WTFMove(values), separator, owner));
    }

    static Ref<DeprecatedCSSOMValueList> create(const CSSValueContainingVector& values, CSSStyleDeclaration& owner)
    {
        return adoptRef(*new DeprecatedCSSOMValueList(values, owner));
    }

    String cssText() const;

    size_t length() const { return m_values.size(); }
    DeprecatedCSSOMValue* item(size_t index) { return index < m_values.size() ? m_values[index].ptr() : nullptr; }
    bool isSupportedPropertyIndex(unsigned index) const { return index < m_values.size(); }

private:
    DeprecatedCSSOMValueList(Vector<Ref<DeprecatedCSSOMValue>, 4> values, CSSValue::ValueSeparator separator, CSSStyleDeclaration& owner)
        : DeprecatedCSSOMValue(ClassType::List, owner)
        , m_values { WTFMove(values) }
    {
        m_valueSeparator = separator;
    }

    DeprecatedCSSOMValueList(const CSSValueContainingVector& values, CSSStyleDeclaration& owner)
        : DeprecatedCSSOMValue(ClassType::List, owner)
        , m_values(WTF::map(values, [&](auto& value) { return value.createDeprecatedCSSOMWrapper(owner); }))
    {
        m_valueSeparator = values.separator();
    }

    Vector<Ref<DeprecatedCSSOMValue>, 4> m_values;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSSOM_VALUE(DeprecatedCSSOMValueList, isValueList())
