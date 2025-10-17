/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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

#include "CSSValue.h"

#include <wtf/text/WTFString.h>

namespace WebCore {

class Element;

class CSSAttrValue final : public CSSValue {
public:
    static Ref<CSSAttrValue> create(String attributeName, RefPtr<CSSValue>&& fallback = nullptr);
    const String attributeName() const { return m_attributeName; }
    const CSSValue* fallback() const { return m_fallback.get(); }
    bool equals(const CSSAttrValue& other) const;
    String customCSSText() const;

private:
    explicit CSSAttrValue(String&& attributeName, RefPtr<CSSValue>&& fallback)
        : CSSValue(ClassType::Attr)
        , m_attributeName(WTFMove(attributeName))
        , m_fallback(WTFMove(fallback))
    {
    }

    String m_attributeName;
    RefPtr<CSSValue> m_fallback;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSAttrValue, isAttrValue())
